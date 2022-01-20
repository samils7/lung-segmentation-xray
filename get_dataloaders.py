import torchvision.transforms as transforms
from custom_transforms import GrayScale, Resize, ToTensor, histogram_equalize, gamma_correction, channel_wise
from lungsegdatasets import chn_dataset, mcu_dataset
from torch.utils.data import DataLoader


def get_dataloaders(args):
    img_size = args.img_size

    if args.aug_method == "gamma":
        custom_transforms = transforms.Compose([
            GrayScale(),
            Resize(img_size),
            histogram_equalize(),
            gamma_correction(0.5),
            ToTensor(),
        ])
    elif args.aug_method == "no_gamma":
        custom_transforms = transforms.Compose([
            GrayScale(),
            Resize(img_size),
            ToTensor(),
        ])
    elif args.aug_method == "channel_wise":
        custom_transforms = transforms.Compose([
            GrayScale(),
            Resize(img_size),
            channel_wise(0.5),
            ToTensor(),
        ])
    else:
        raise NotImplementedError(f"{args.aug_method} is not implemented")

    chn_train = chn_dataset(split='train', transforms=custom_transforms)
    chn_val = chn_dataset(split='val', transforms=custom_transforms)
    mcu_train = mcu_dataset(split='train', transforms=custom_transforms)
    mcu_val = mcu_dataset(split='val', transforms=custom_transforms)

    dataloader = {'train': {
        'chn': DataLoader(dataset=chn_train, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True),
        'mcu': DataLoader(dataset=mcu_train, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)},
        'val': {'chn': DataLoader(dataset=chn_val, batch_size=args.batch_size, num_workers=args.n_workers),
                'mcu': DataLoader(dataset=mcu_val, batch_size=args.batch_size, num_workers=args.n_workers)}}

    return dataloader
