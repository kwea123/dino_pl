import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of images')

    # model parameters
    parser.add_argument('--arch', default='small', type=str,
                        choices=['tiny', 'small', 'base'],
                        help="which vit to train.")
    parser.add_argument('--patch_size', default=16, type=int,
                        help="""Size in pixels of input square patches - default 16 (for 16x16 patches).
                        Using smaller values leads to better performance but requires more memory.
                        Applies only for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
                        mixed precision training to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int,
                        help="""Dimensionality of the DINO head output.
                        For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                        help="stochastic depth rate")
    parser.add_argument('--norm_last_layer', default=False, action="store_true",
                        help="""Whether or not to weight normalize the last layer of the DINO head.
                        Not normalizing leads to better performance but can make the training unstable.
                        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', type=float, default=0.9995,
                        help="""Base EMA parameter for teacher update.
                        The value is increased to 1 during training with cosine schedule.
                        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")

    # augmentation parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relative to the origin image.
                        Used for large global view cropping.""")
    parser.add_argument('--local_crops_number', type=int, default=8,
                        help="""Number of small local views to generate.""")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relative to the origin image.
                        Used for small local view cropping of multi-crop.""")

    # temperature parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
                        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--final_teacher_temp', default=0.07, type=float,
                        help="""Final value (after linear warmup) of the teacher temperature.
                        For most experiments, anything above 0.07 is unstable. We recommend
                        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature.')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size per gpu')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    
    parser.add_argument('--weight_decay_init', type=float, default=0.04,
                        help="""Initial value of the weight decay.
                        With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4,
                        help="""Final value of the weight decay.
                        We use a cosine schedule for WD and using a larger decay by
                        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0,
                        help="""Clipping with norm .3 ~ 1.0 can
                        help optimization for larger ViT architectures. 0 for disabling.""")

    # parser.add_argument('--ckpt_path', type=str, default=None,
    #                     help='pretrained checkpoint path to load')

    parser.add_argument('--fp16', default=False, action='store_true',
                        help='use fp16 training')
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate at the end of warmup (for batch size=256)')
    parser.add_argument('--ep_freeze_last_layer', default=1, type=int,
                        help="""number of epochs during which we keep the output layer fixed.
                        Typically doing so during the first epoch helps training.
                        Try increasing this value if the loss does not decrease.""")

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()