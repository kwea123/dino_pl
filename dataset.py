from torch.utils.data import Dataset
import os
import glob
import imageio
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, root, split='train'):
        self.split = split
        self.image_paths = []
        # TODO: save image paths to file to avoid reading overhead
        stamp_idxs = sorted(os.listdir(root))
        print('loading image paths ...')
        for stamp_idx in stamp_idxs:
            image_paths = glob.glob(os.path.join(root, f'{stamp_idx}/[0-9]*.png'))
            image_paths = sorted(filter(lambda x: not 'key' in x, image_paths))
            self.image_paths += image_paths
        print(f'{len(self.image_paths)} image paths loaded!')

    def __len__(self):
        if self.split == 'train':
            return len(self.image_paths)
        return 1

    def __getitem__(self, idx):
        if self.split != 'train': # randomly choose an image for validation
            idx = np.random.choice(len(self.image_paths), 1)[0]
        image = imageio.imread(self.image_paths[idx])
        if image.shape[-1] == 4: # if there is alpha channel
            image[image[..., -1]==0, :3]= 255 # a=0 to white
            image = image[..., :3]
        return self.transform(image)


class ValTransform:
    def __init__(self):
        self.norm = A.Compose([
            A.Resize(224, 224, interpolation=cv2.INTER_AREA),
            A.Normalize(),
            ToTensorV2()
        ])

        self.orig = A.Compose([
            A.Resize(224, 224, interpolation=cv2.INTER_AREA),
            ToTensorV2()
        ])

    def __call__(self, image):
        return [self.orig(image=image)['image'], self.norm(image=image)['image']]


class TrainTransform:
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            A.ToGray(p=0.2)
        ])
        normalize = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])

        # area interpolation should be better for small image
        # first global crop
        self.global_crop1 = A.Compose([
            A.RandomResizedCrop(224, 224, scale=global_crops_scale, interpolation=cv2.INTER_AREA),
            flip_and_color_jitter,
            A.GaussianBlur(p=1.0),
            normalize,
        ])
        # second global crop
        self.global_crop2 = A.Compose([
            A.RandomResizedCrop(224, 224, scale=global_crops_scale, interpolation=cv2.INTER_AREA),
            flip_and_color_jitter,
            A.GaussianBlur(p=0.1),
            A.Solarize(p=0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_crop = A.Compose([
            A.RandomResizedCrop(96, 96, scale=local_crops_scale, interpolation=cv2.INTER_AREA),
            flip_and_color_jitter,
            A.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = [self.global_crop1(image=image)['image'], self.global_crop2(image=image)['image']]
        for _ in range(self.local_crops_number):
            crops += [self.local_crop(image=image)['image']]
        return crops