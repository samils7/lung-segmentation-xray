from cProfile import label
import os
import torch
import matplotlib.pyplot as plt
from model.unet import UNet
from get_dataloaders import get_dataloaders
import argparse
import numpy as np


def plot(images, masks, predictions, accs, ious, title, name):
    
    plt.rcParams['figure.figsize'] = [15, 15]
    plt.figure()
    
    length = len(images)
    i = 1
    col_size = 3
    for image, mask, prediction, acc, iou in zip(images, masks, predictions, accs, ious):
        plt.subplot(length, col_size, i)
        plt.imshow(image, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")
        i += 1

        plt.subplot(length, col_size, i)
        plt.imshow(mask, cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")
        i += 1

        
        plt.subplot(length, col_size, i)
        plt.imshow(prediction, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")
        plt.title(f"IoU: {iou:.5f}\nAcc{acc:.5f}")
        i += 1
    
    plt.suptitle(title)
    plt.savefig(f"{name}.png")
    plt.savefig(f"{name}.pdf")




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="U-Net for Lung Segmentation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    args.img_size = 512
    args.batch_size = 1
    args.n_workers = 12
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # default: '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    base = "results"
    for ciriterion in os.listdir(f"{base}"):
        for augmentation in os.listdir(f"{base}/{ciriterion}"):
            args.aug_method = augmentation
            args.criterion = ciriterion
            dataloader = get_dataloaders(args)
            dataloader = dataloader["val"]["chn"]

            path = f"{base}/{ciriterion}/{augmentation}/best_checkpoint.pt"
            checkpoint = torch.load(path, map_location=device)
            model = UNet(n_channels=1 if augmentation != "channel_wise" else 3, n_classes=1, device=device)
            model.load_state_dict(checkpoint['model_state_dict'])

            images = []
            masks = []
            predictions = []
            accs = []
            ious = []
            for batch_idx, sample in enumerate(dataloader):
                imgs, true_masks = sample['image'], sample['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32).round()
                with torch.set_grad_enabled(False):
                    masks_pred = model(imgs)
                        
                    pred = torch.sigmoid(masks_pred).round().detach()
                    images.append(imgs.cpu().numpy()[0][0])
                    masks.append(true_masks.cpu().numpy()[0][0])
                    predictions.append(pred.cpu().numpy()[0][0])
                    accs.append((pred == true_masks).float().mean().item())
                    ious.append((torch.logical_and(pred, true_masks).sum() / torch.logical_or(pred, true_masks).sum()).item())
                    

                if batch_idx == 3:
                    break
            
            plot(images, masks, predictions, accs, ious, f"{ciriterion} {augmentation}", f"{base}/{ciriterion}/{augmentation}/inference")

        
