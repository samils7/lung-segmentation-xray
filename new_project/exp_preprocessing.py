import torchvision.transforms as transforms
from custom_transforms import GrayScale, Resize, ToTensor, histogram_equalize, gamma_correction
from lungsegdatasets import chn_dataset, mcu_dataset
from torch.utils.data import DataLoader
from numpy import dtype
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def plot(data:dict, name:str):
    plt.rcParams['figure.figsize'] = [20, 10]
    length = len(data)

    for i, title in enumerate(data):
        plt.subplot(1, length, i + 1)

        plt.imshow(data[title], cmap="gray")


        plt.title(title)
        plt.axis("off")
    
    plt.savefig(name)



if __name__ == "__main__":
    image_path = "data/Lung Segmentation/CXR_png/CHNCXR_0002_0.png"
    imgs = {}
    img = Image.open(image_path).convert("LA")
    imgs["Original Image"] = img.copy()
    img = GrayScale()({"image": img.copy()})["image"]
    imgs["GrayScale"] = img.copy()
    img = histogram_equalize()({"image": img.copy()})["image"]
    imgs["Histogram Equalized"] = img.copy()
    img = gamma_correction(0.5)({"image": img.copy()})["image"]
    imgs["Gamma Correction"] = img.copy()

    plot(imgs, "preprocess_with_gamma.png")

    image_path = "data/Lung Segmentation/CXR_png/CHNCXR_0002_0.png"
    imgs = {}
    img = Image.open(image_path).convert("LA")
    imgs["Original Image"] = img.copy()
    img = GrayScale()({"image": img.copy()})["image"]
    imgs["GrayScale"] = img.copy()

    plot(imgs, "preprocess_with_no_gamma.png")
