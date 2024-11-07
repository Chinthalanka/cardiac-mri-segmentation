import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            # images, msks_lv, msks_rv, msks_myo = batch['image'], batch['mask_lv'], batch['mask_rv'], batch['mask_myo']
            images, masks, masks_all = batch['image'], batch['masks'], batch['masks_all']
            images = images.permute(0, 3, 1, 2)
            masks_all = masks_all.permute(0, 3, 1, 2)

            # move images and labels to correct device and type
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = masks.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            # mask_true = masks_all.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            '''
            mask_true = torch.stack(
                [torch.tensor(msks_lv, dtype=torch.float32),
                 torch.tensor(msks_rv, dtype=torch.float32),
                 torch.tensor(msks_myo, dtype=torch.float32)],
                dim=0).to(device=device, dtype=torch.float32, memory_format=torch.channels_last).permute(1, 0, 2, 3)
            '''

            # predict the mask
            # mask_pred = net(images).permute(1, 0, 2, 3)
            mask_pred = net(images)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred.squeeze(1), mask_true.float(), reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
                # convert to one-hot format
                # mask_true = mask_true.long()
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # mask_pred = mask_pred.float()

                # Compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, :], mask_true[:, :], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)
