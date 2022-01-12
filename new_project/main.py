import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim

from model.unet import UNet
from get_dataloaders import get_dataloaders
from DiceLoss import DiceLoss
from utils import Logger
from train import train, eval


def main(args, device):
    model = UNet(n_channels=1, n_classes=1, device=device)

    """load checkpoint pt"""
    start_epoch = 0
    if args.load_model:
        print("load model from", args.load_model)
        checkpoint = torch.load(args.load_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']

    dataloader = get_dataloaders(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5)

    if args.criterion == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif args.criterion == "dice":
        criterion = DiceLoss()
    else:
        raise NotImplementedError(f"{args.criterion} is not implemented")

    logger = Logger(args)
    logger(f"args: {args}")
    epochs = args.epochs
    train(model, criterion, optimizer, scheduler, dataloader, device, epochs, logger, start_epoch)
    checkpoint = torch.load(f"{logger.path}/best_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    eval(model, criterion, dataloader, device, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net for Lung Segmentation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--n_workers', type=int, default=6, help="The number of workers for dataloader")

    # arguments for training
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.1)

    parser.add_argument('--aug_method', type=str, default='gamma', choices=['gamma', 'no_gamma'])
    parser.add_argument('--criterion', type=str, default='BCE', choices=['BCE', 'dice'])

    parser.add_argument('--load_model', type=str, default=None, help='.pth file path to load model')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # default: '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args, device)
