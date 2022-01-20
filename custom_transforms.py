import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np


class GrayScale(object):
    def __call__(self, sample):
        from torchvision.transforms import Grayscale
        Grayscale = Grayscale()
        sample['image'] = Grayscale(sample['image'])
        return sample


class histogram_equalize(object):
    def __call__(self, sample):
        sample['image'] = ImageOps.equalize(sample['image'], mask=None)
        return sample


class gamma_correction(object):
    def __init__(self, gamma):
        assert isinstance(gamma, (int, float))
        self.gamma = gamma

    def __call__(self, sample):
        im = np.array(sample['image'])
        im_g = 255.0 * (im / 255.0) ** (1.0 / self.gamma)
        pil_img = Image.fromarray(np.uint8(im_g))
        sample['image'] = pil_img
        return sample


class channel_wise(object):
    def __init__(self, gamma):
        assert isinstance(gamma, (int, float))
        self.gamma_correction = gamma_correction(gamma)

    def __call__(self, sample):
        im_0 = np.array(sample['image'])
        im_1 = histogram_equalize()(sample)['image']
        im_2 = self.gamma_correction(sample)['image']
        sample['image'] = np.stack((im_0, im_1, im_2), -1)

        return sample


class Resize(object):
    """
    Resize the input PIL Image to the given size.
    """

    def __init__(self, img_size):
        assert isinstance(img_size, (int, tuple))
        self.img_size = img_size

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        Resize = transforms.Resize((self.img_size, self.img_size))
        sample['image'], sample['mask'] = Resize(img), Resize(mask)
        return sample


class ToTensor(object):
    """convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        from torchvision.transforms import ToTensor
        ToTensor = ToTensor()
        img, mask = sample['image'], sample['mask']
        sample['image'], sample['mask'] = ToTensor(img), ToTensor(mask)
        return sample


class ToPILImage(object):
    def __call__(self, sample):
        from torchvision.transforms import ToPILImage
        img, mask = sample['image'], sample['mask']
        ToPILImage = ToPILImage()
        sample['image'], sample['mask'] = ToPILImage(img), ToPILImage(mask)
        return sample
