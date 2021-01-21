import torch
import numpy as np


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, imgs):
        for i, img in enumerate(imgs):
            img = np.array(img).astype(np.float32)
            img /= 255.0
            img -= self.mean
            img /= self.std
            imgs[i] = img

        return imgs


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, imgs):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        for i, img in enumerate(imgs):
            imgs[i] = np.array(img).astype(np.float32).transpose((2, 0, 1))
        imgs = np.vstack(imgs)
        imgs = torch.from_numpy(imgs).float()

        return imgs
