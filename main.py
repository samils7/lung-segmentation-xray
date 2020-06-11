import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from lungsegdatasets import chn_dataset, mcu_dataset
import argparse
import os

from torch import optim
from tqdm import tqdm
from model.unet import UNet
import time
import copy


def train_and_validate(net, criterion, optimizer, scheduler, dataloader, device, epochs, load_model=None):
    """load checkpoint pt"""
    if load_model:
        print("load model from", load_model)
        checkpoint = torch.load(load_model)
        net.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    history = {'train': {'epoch': [], 'loss': [], 'acc': []},
               'val': {'epoch': [], 'loss': [], 'acc': []}}

    best_acc = 0.90
    start = time.time()
    for epoch in range(epochs):
        if load_model and epoch == 0:
            epoch += start_epoch
            epochs += start_epoch
        print("-" * 30)
        print(f"Epoch {epoch + 1}/{epochs}")
        # print(f"Epoch {epoch + 1}/{epochs} learning_rate : {scheduler.get_lr()[0]}")

        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # set model to training mode
            else:
                print("-" * 10)
                net.eval()  # set model to evaluate mode

            running_loss = 0.0
            running_correct = 0
            dataset_size = 0
            """Iterate over data"""
            for dset in dataloader[phase]:
                for batch_idx, sample in enumerate(tqdm(dataloader[phase][dset])):
                    imgs, true_masks = sample['image'], sample['mask']
                    imgs = imgs.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.float32)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    """forward"""
                    with torch.set_grad_enabled(phase == 'train'):
                        masks_pred = net(imgs)
                        loss = criterion(masks_pred, true_masks)
                        running_loss += loss.item()

                        """backward + optimize only if in training phase"""
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    """ statistics """
                    dataset_size += imgs.size(0)
                    running_loss += loss.item() * imgs.size(0)
                    pred = torch.sigmoid(masks_pred) > 0.5
                    running_correct += (pred == true_masks).float().mean().item() * imgs.size(0)
                    # running_acc = running_correct / dataset_size
                    # if (batch_idx + 1) % 40 == 0:
                    # print(f'Batch {batch_idx}/{len(dataloader[phase])} Loss {loss.item()} Acc {running_acc}')
                    # print(f'Batch {batch_idx+1}/{len(dataloader[phase])} Loss {loss.item()}')

            """ statistics """
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_correct / dataset_size
            print('{} Loss {:.5f}\n{} Acc {:.2f}'.format(phase, epoch_loss, phase, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())
                torch.save({'epoch': epoch + 1,
                            'model_state_dict': best_model_wts,
                            'optimizer_state_dict': optimizer.state_dict()
                            }, os.path.join(os.getcwd(), 'checkpoint/best_checkpoint[epoch_{}].pt'.format(epoch + 1)))
                print("Achieved best result! save checkpoint.")
                print("val acc = ", best_acc)
            history[phase]['epoch'].append(epoch)
            history[phase]['loss'].append(epoch_loss)
            history[phase]['acc'].append(epoch_acc)

        scheduler.step(history['val']['acc'][-1])

        time_elapsed = time.time() - since
        print("One Epoch Complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        time_elapsed = time.time() - start
        minute, sec = time_elapsed // 60, time_elapsed % 60
        print("Total Training time {:.0f}min {:.0f}sec".format(minute, sec))


def get_args():
    parser = argparse.ArgumentParser(description="U-Net for Lung Segmentation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--n_workers', type=int, default=4, help="The number of workers for dataloader")

    # arguments for training
    parser.add_argument('--img_size', type=int, default=512, )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.1)

    parser.add_argument('--load_model', type=str, default=None, help='.pth file path to load model')
    return parser.parse_args()


def main():
    args = get_args()

    # set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # default: '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set model
    model = UNet(n_channels=1, n_classes=1).to(device)
    if len(args.gpu) > 1:  # if multi-gpu
        model = torch.nn.DataParallel(model)

    img_size = args.img_size  # default: 512
    # set transforms for dataset
    import torchvision.transforms as transforms
    from my_transforms import GrayScale, Resize, ToTensor, histogram_equalize, gamma_correction
    custom_transforms = transforms.Compose([
        GrayScale(),
        Resize(img_size),
        histogram_equalize(),
        gamma_correction(0.5),
        ToTensor(),
    ])

    # set Dataset and DataLoader
    chn_train = chn_dataset(split='train', transforms=custom_transforms)
    chn_val = chn_dataset(split='val', transforms=custom_transforms)
    mcu_train = mcu_dataset(split='train', transforms=custom_transforms)
    mcu_val = mcu_dataset(split='val', transforms=custom_transforms)

    from torch.utils.data import DataLoader
    dataloader = {'train': {'chn': DataLoader(dataset=chn_train, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True),
                            'mcu': DataLoader(dataset=mcu_train, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)},
                  'val':   {'chn': DataLoader(dataset=chn_val, batch_size=args.batch_size, num_workers=args.n_workers),
                            'mcu': DataLoader(dataset=mcu_val, batch_size=args.batch_size, num_workers=args.n_workers)}}

    # checkpoint dir
    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoint')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_path = args.load_model

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # learning rate scheduler
    # from torch.optim.lr_scheduler import StepLR
    # scheduler = StepLR(optimizer, step_size = 3 , gamma = 0.8)
    # option 2.
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5)

    criterion = nn.BCEWithLogitsLoss()

    train_and_validate(net=model, criterion=criterion, optimizer=optimizer, dataloader=dataloader, device=device,
                       epochs=args.epochs, scheduler=scheduler, load_model=checkpoint_path)


if __name__ == '__main__':
    main()
