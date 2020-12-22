import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
from dataloaders.utils import encode_segmap
import torchvision.transforms.functional as t_func


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        imgs = sample['image']
        for i, img in enumerate(imgs):
            img = np.array(img).astype(np.float32)
            img /= 255.0
            img -= self.mean
            img /= self.std
            imgs[i] = img

        mask = sample['label']
        mask = np.array(mask).astype(np.float32)
        mask = encode_segmap(mask)

        return {'image': imgs,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        imgs = sample['image']
        mask = sample['label']
        for i, img in enumerate(imgs):
            imgs[i] = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)
        imgs = np.vstack(imgs)

        img = torch.from_numpy(imgs).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        imgs = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            for i, img in enumerate(imgs):
                imgs[i] = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': imgs,
                'label': mask}


# class RandomRotate(object):
#     def __init__(self, degree):
#         self.degree = degree
#
#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         rotate_degree = random.uniform(-1*self.degree, self.degree)
#         img = img.rotate(rotate_degree, Image.BILINEAR)
#         mask = mask.rotate(rotate_degree, Image.NEAREST)
#
#         return {'image': img,
#                 'label': mask}


# class RandomGaussianBlur(object):
#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         if random.random() < 0.5:
#             img = img.filter(ImageFilter.GaussianBlur(
#                 radius=random.random()))
#
#         return {'image': img,
#                 'label': mask}


class RandomCrop(object):
    def __init__(self, crop_size, fill=0):
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        imgs = sample['image']
        mask = sample['label']
        # random crop crop_size
        w, h = imgs[0].size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        for i, img in enumerate(imgs):
            imgs[i] = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': imgs,
                'label': mask}


class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, gamma=0.1):
        self.brightness = random.uniform(1-brightness, 1+brightness)
        self.contrast = random.uniform(1-contrast, 1+contrast)
        self.saturation = random.uniform(1-saturation, 1+saturation)
        self.hue = random.uniform(-hue, hue)
        self.gamma = random.uniform(1-gamma, 1+gamma)

    def __call__(self, sample):
        imgs = sample['image']
        mask = sample['label']
        for i, img in enumerate(imgs):
            img = t_func.adjust_brightness(img, self.brightness)
            img = t_func.adjust_contrast(img, self.contrast)
            img = t_func.adjust_saturation(img, self.saturation)
            img = t_func.adjust_hue(img, self.hue)
            imgs[i] = t_func.adjust_gamma(img, self.gamma)

        return {'image': imgs,
                'label': mask}
