import torch
import numpy as np
from matplotlib import cm
import copy
from torchvision.utils import make_grid

from opt import get_opts

# dataset
from dataset import ImageDataset, TrainTransform, ValTransform
from torch.utils.data import DataLoader

# model
from models import vits_dict, MultiCropWrapper, DINOHead
from losses import DINOLoss

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


def att2img(att, cmap=cm.get_cmap('plasma')):
    """
    att: (H, W)
    """
    x = att.cpu().numpy()
    mi, ma = np.min(x), np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x_ = cmap(x)[..., :3] # (H, W, 3)
    x_ = torch.from_numpy(x_).permute(2, 0, 1) # (3, H, W)
    return x_


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=1e-6):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * \
                             (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class DINOSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False

        model = vits_dict[hparams.arch]
        student_backbone = model(patch_size=hparams.patch_size,
                                 drop_path_rate=hparams.drop_path_rate)
        self.teacher_backbone = model(patch_size=hparams.patch_size)

        student_head = DINOHead(student_backbone.embed_dim, hparams.out_dim,
                                hparams.norm_last_layer)
        teacher_head = DINOHead(self.teacher_backbone.embed_dim, hparams.out_dim)

        self.student = MultiCropWrapper(student_backbone, student_head)
        if hparams.pretrained_path: # fine-tune from pretrained dino
            print(f'loading pretrained model from {hparams.pretrained_path} ...')
            ckpt = torch.load(hparams.pretrained_path, map_location='cpu')
            self.student.load_state_dict(ckpt['teacher'])
        self.teacher = MultiCropWrapper(self.teacher_backbone, teacher_head)
        # teacher and student start with the same weights
        self.teacher.load_state_dict(self.student.state_dict())

        # teacher is not trained
        for p in self.teacher.parameters(): p.requires_grad = False

        self.loss = DINOLoss(hparams.out_dim,
                             hparams.local_crops_number+2,
                             hparams.warmup_teacher_temp,
                             hparams.final_teacher_temp,
                             hparams.warmup_teacher_temp_epochs,
                             hparams.num_epochs)

    def setup(self, stage=None):
        print('loading image paths ...')
        self.train_dataset = ImageDataset(hparams.root_dir, 'train')
        print(f'{len(self.train_dataset.image_paths)} image paths loaded!')

        self.val_dataset = copy.deepcopy(self.train_dataset)
        self.val_dataset.split = 'val'

        self.train_dataset.transform = \
            TrainTransform(hparams.global_crops_scale,
                           hparams.local_crops_scale,
                           hparams.local_crops_number)
        self.val_dataset.transform = ValTransform()

    def configure_optimizers(self):
        regularized, not_regularized = [], []
        for n, p in self.student.named_parameters():
            if not p.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if n.endswith(".bias") or len(p.shape) == 1:
                not_regularized.append(p)
            else:
                regularized.append(p)
        param_groups = [{'params': regularized},
                        {'params': not_regularized, 'weight_decay': 0.}]

        self.lr = hparams.lr * (hparams.batch_size*hparams.num_gpus/256)
        opt = torch.optim.AdamW(param_groups, self.lr)

        return opt

    def train_dataloader(self):
        self.loader = DataLoader(self.train_dataset,
                                 shuffle=True,
                                 num_workers=hparams.num_workers,
                                 batch_size=hparams.batch_size,
                                 pin_memory=True,
                                 drop_last=True)

        # define schedulers based on number of iterations
        niter_per_ep = len(self.loader)
        self.lr_sch = cosine_scheduler(self.lr, 1e-6, hparams.num_epochs, niter_per_ep//hparams.num_gpus,
                                       hparams.warmup_epochs)
        # weight decay scheduler
        self.wd_sch = cosine_scheduler(hparams.weight_decay_init, hparams.weight_decay_end,
                                       hparams.num_epochs, niter_per_ep//hparams.num_gpus)
        # momentum scheduler
        self.mm_sch = cosine_scheduler(hparams.momentum_teacher, 1.0,
                                       hparams.num_epochs, niter_per_ep//hparams.num_gpus)

        return self.loader

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=hparams.num_workers,
                          batch_size=1, # validate one image
                          pin_memory=True)

    def training_step(self, batch, batch_idx):
        """
        batch: a list of "2+local_crops_number" tensors
               each tensor is of shape (B, 3, h, w)
        """
        opt = self.optimizers()
        # update learning rate, weight decay
        for i, param_group in enumerate(opt.param_groups):
            param_group['lr'] = self.lr_sch[self.global_step]
            if i == 0: # only the first group is regularized
                param_group['weight_decay'] = self.wd_sch[self.global_step]

        teacher_output = self.teacher(batch[:2])
        student_output = self.student(batch)
        loss = self.loss(student_output, teacher_output, self.current_epoch)

        opt.zero_grad()
        self.manual_backward(loss)
        # clip gradient
        if hparams.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), hparams.clip_grad)
        # cancel gradient for the first epochs
        if self.current_epoch < hparams.ep_freeze_last_layer:
            for n, p in self.student.named_parameters():
                if "last_layer" in n:
                    p.grad = None
        opt.step()

        # EMA update for the teacher
        m = self.mm_sch[self.global_step]
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_((1-m)*ps.data)

        self.log('rates/lr', opt.param_groups[0]['lr'])
        self.log('rates/weight_decay', opt.param_groups[0]['weight_decay'])
        self.log('rates/momentum', m)
        self.log('train/loss', loss, True)

    def validation_step(self, batch, batch_idx):
        img_orig, img_norm = batch

        w_featmap = img_norm.shape[-1] // hparams.patch_size
        h_featmap = img_norm.shape[-2] // hparams.patch_size

        atts = self.teacher_backbone.get_last_selfattention(img_norm)
        atts = atts[:, :, 0, 1:].reshape(1, -1, h_featmap, w_featmap)
        atts = torch.nn.functional.interpolate(atts,
                    scale_factor=hparams.patch_size, mode="nearest")[0] # (6, h, w)

        return {'attentions': atts, 'img': img_orig}

    def validation_epoch_end(self, outputs):
        atts = outputs[0]['attentions']

        tb = self.logger.experiment
        tb.add_image('image', outputs[0]['img'][0], self.global_step)
        atts_vis = [att2img(att) for att in atts]
        tb.add_image('attentions', make_grid(atts_vis, nrow=3), self.global_step)


if __name__ == '__main__':
    hparams = get_opts()
    system = DINOSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_top_k=-1)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      precision=16 if hparams.fp16 else 32,
                      accelerator='auto',
                      devices=hparams.num_gpus,
                      strategy=DDPStrategy(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=1)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)