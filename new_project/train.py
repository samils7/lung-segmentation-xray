import torch
from tqdm import tqdm
import time
import copy


def train(model, criterion, optimizer, scheduler, dataloader, device, epochs, logger, start_epoch=0):
    history = {'train': {'epoch': [], 'loss': [], 'acc': [], 'iou': []},
               'val': {'epoch': [], 'loss': [], 'acc': [], 'iou': []}}

    best_iou = -1.0
    start = time.time()
    for epoch in tqdm(range(start_epoch, epochs + start_epoch)):
        logger("-" * 30)
        logger(f"Epoch {epoch + 1}/{epochs}")
        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                logger("-" * 10)
                model.eval()  # set model to evaluate mode

            running_loss = 0.0
            running_correct = 0
            running_iou = 0.0
            dataset_size = 0
            """Iterate over data"""
            for dset in tqdm(dataloader[phase], disable=True):
                for batch_idx, sample in enumerate(tqdm(dataloader[phase][dset])):
                    imgs, true_masks = sample['image'], sample['mask']
                    imgs = imgs.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.float32)

                    optimizer.zero_grad()

                    """forward"""
                    with torch.set_grad_enabled(phase == 'train'):
                        masks_pred = model(imgs)
                        loss = criterion(masks_pred, true_masks)
                        running_loss += loss.item()

                        """backward + optimize only if in training phase"""
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    """ statistics """
                    dataset_size += imgs.size(0)
                    running_loss += loss.item() * imgs.size(0)
                    pred = torch.sigmoid(masks_pred).round()
                    running_correct += (pred == true_masks).float().mean().item() * imgs.size(0)
                    running_iou += (torch.logical_and(pred, true_masks).sum() / torch.logical_or(pred,
                                                                                                 true_masks).sum()) * imgs.size(
                        0)

            """ statistics """
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_correct / dataset_size
            epoch_iou = running_iou / dataset_size
            logger('{} Loss {:.5f}\n{} Acc {:.2f} IoU {:.2f}'.format(phase, epoch_loss, phase, epoch_acc, epoch_iou))
            if phase == 'val' and epoch_iou > best_iou:
                best_iou = epoch_iou
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({'epoch': epoch,
                            'model_state_dict': best_model_wts
                            }, f'{logger.path}/best_checkpoint.pt')
                logger("Achieved best result! save checkpoint.")
                logger(
                    '{} Loss {:.5f}\n{} Acc {:.2f} IoU {:.2f}'.format(phase, epoch_loss, phase, epoch_acc, epoch_iou))

            history[phase]['epoch'].append(epoch)
            history[phase]['loss'].append(epoch_loss)
            history[phase]['acc'].append(epoch_acc)
            history[phase]['iou'].append(epoch_iou)

        scheduler.step(history['val']['loss'][-1])

        time_elapsed = time.time() - since
        logger("One Epoch Complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        time_elapsed = time.time() - start
        minute, sec = time_elapsed // 60, time_elapsed % 60
        logger("Total Training time {:.0f}min {:.0f}sec".format(minute, sec))

        final_wts = copy.deepcopy(model.state_dict())
        torch.save({'epoch': epoch,
                    'model_state_dict': final_wts
                    }, f'{logger.path}/last_checkpoint.pt')


def eval(model, criterion, dataloader, device, logger):
    model.eval()  # set model to evaluate mode

    running_loss = 0.0
    running_correct = 0
    running_iou = 0.0
    dataset_size = 0
    """Iterate over data"""
    for dset in tqdm(dataloader['val'], disable=True):
        for batch_idx, sample in enumerate(tqdm(dataloader['val'][dset])):
            imgs, true_masks = sample['image'], sample['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            with torch.set_grad_enabled(False):
                masks_pred = model(imgs)
                loss = criterion(masks_pred, true_masks)
                running_loss += loss.item()

            """ statistics """
            dataset_size += imgs.size(0)
            running_loss += loss.item() * imgs.size(0)
            pred = torch.sigmoid(masks_pred).round()
            running_correct += (pred == true_masks).float().mean().item() * imgs.size(0)
            running_iou += (torch.logical_and(pred, true_masks).sum() / torch.logical_or(pred,
                                                                                         true_masks).sum()) * imgs.size(
                0)

    """ statistics """
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_correct / dataset_size
    epoch_iou = running_iou / dataset_size
    logger('{} Loss {:.5f}\n{} Acc {:.2f} IoU {:.2f}'.format('Test', epoch_loss, 'Test', epoch_acc, epoch_iou))

