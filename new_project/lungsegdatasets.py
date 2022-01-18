from numpy import dtype
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class chn_dataset(Dataset):  # inherit from torch.utils.data.Dataset
    """China dataset."""

    def __init__(self, root_dir=os.path.join(os.getcwd(), "data/Lung Segmentation"), split="train", transforms=None,
                 shuffle=True):
        """
        Args:
        :param root_dir (str):
        :param split (str):
        :param transforms (callable, optional) :
        """
        self.root_dir = root_dir
        self.split = split  # train / val
        self.transforms = transforms

        # images
        self.image_path = self.root_dir + '/CXR_png'
        image_file = os.listdir(self.image_path)
        self.chn_image_files = [fName for fName in image_file if "CHNCXR" in fName]
        self.chn_image_idx = sorted([int(fName.split("_")[1]) for fName in self.chn_image_files])

        # masks
        self.mask_path = os.path.join(self.root_dir, 'masks')
        mask_file = os.listdir(self.mask_path)
        self.chn_mask_files = [fName for fName in mask_file if "CHNCXR" in fName]
        self.chn_mask_idx = sorted([int(fName.split("_")[1]) for fName in self.chn_mask_files])

        # train/ val
        self.all_idx = [idx for idx in self.chn_image_idx if idx in self.chn_mask_idx]
        self.train_idx = self.all_idx[:int(0.8 * len(self.all_idx))]
        self.val_idx = self.all_idx[int(0.8 * len(self.all_idx)):]

        self.data_file = {"image": self.chn_image_files, "mask": self.chn_mask_files}
        self.data_idx = {"train": self.train_idx, "val": self.val_idx}

        # print("The Total number of data =",len(self.train_idx) + len(self.val_idx) + len(self.test_idx))
        # print("The Total number of train data =", len(self.train_idx))
        # print("The Total number of val data =", len(self.val_idx))
        # print("The Total number of test data =", len(self.test_idx))

    def __len__(self):
        return len(self.data_idx[self.split])

    def __getitem__(self, idx):
        idx = self.data_idx[self.split][idx]
        # set index
        for fName in self.data_file["image"]:
            file_idx = int(fName.split('_')[1])
            if idx == file_idx:
                img_fName = fName

        img_path = os.path.join(self.image_path, img_fName)
        img = Image.open(img_path).convert('LA')  # open as PIL Image and set Channel = 1

        for fName in self.data_file["mask"]:
            file_idx = int(fName.split('_')[1])
            if idx == file_idx:
                mask_fName = fName

        mask_path = os.path.join(self.mask_path, mask_fName)
        mask = Image.open(mask_path)  # PIL Image
        mask = Image.fromarray(np.array(mask, dtype=bool))

        sample = {'image': img, 'mask': mask}

        if self.transforms:
            sample = self.transforms(sample)

        if isinstance(img, torch.Tensor) and isinstance(mask, torch.Tensor):
            assert img.size == mask.size
        return sample


class mcu_dataset(Dataset):
    """Montgomery dataset."""

    def __init__(self, root_dir=os.path.join(os.getcwd(), "data/Lung Segmentation"), split="train", transforms=None,
                 shuffle=True):
        """
        Args:
        :param root_dir (str):
        :param split (str):
        :param transforms (callable, optional) :
        """
        self.root_dir = root_dir
        self.split = split  # train / val
        self.transforms = transforms

        # images
        self.image_path = self.root_dir + '/CXR_png'
        image_file = os.listdir(self.image_path)
        self.mcu_image_files = [fName for fName in image_file if "MCUCXR" in fName]
        self.mcu_image_idx = sorted([int(fName.split("_")[1]) for fName in self.mcu_image_files])

        # masks
        self.mask_path = os.path.join(self.root_dir, 'masks')
        mask_file = os.listdir(self.mask_path)
        self.mcu_mask_files = [fName for fName in mask_file if "MCUCXR" in fName]
        self.mcu_mask_idx = sorted([int(fName.split("_")[1]) for fName in self.mcu_mask_files])

        # train/ val
        self.all_idx = [idx for idx in self.mcu_image_idx if idx in self.mcu_mask_idx]
        self.train_idx = self.all_idx[:int(0.8 * len(self.all_idx))]
        self.val_idx = self.all_idx[int(0.8 * len(self.all_idx)):]

        self.data_file = {"image": self.mcu_image_files, "mask": self.mcu_mask_files}
        self.data_idx = {"train": self.train_idx, "val": self.val_idx}

        # print("The Total number of data =",len(self.train_idx) + len(self.val_idx) + len(self.test_idx))
        # print("The Total number of train data =", len(self.train_idx))
        # print("The Total number of val data =", len(self.val_idx))
        # print("The Total number of test data =", len(self.test_idx))

    def __len__(self):
        return len(self.data_idx[self.split])

    def __getitem__(self, idx):
        idx = self.data_idx[self.split][idx]
        # set index
        for fName in self.data_file["image"]:
            file_idx = int(fName.split('_')[1])
            if idx == file_idx:
                img_fName = fName

        img_path = os.path.join(self.image_path, img_fName)
        img = Image.open(img_path).convert('LA')  # open as PIL Image and set Channel = 1

        for fName in self.data_file["mask"]:
            file_idx = int(fName.split('_')[1])
            if idx == file_idx:
                mask_fName = fName

        mask_path = os.path.join(self.mask_path, mask_fName)
        mask = Image.open(mask_path)  # PIL Image
        mask = Image.fromarray(np.array(mask, dtype=bool))

        sample = {'image': img, 'mask': mask}

        if self.transforms:
            sample = self.transforms(sample)

        if isinstance(img, torch.Tensor) and isinstance(mask, torch.Tensor):
            assert img.size == mask.size
        return sample
