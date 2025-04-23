"""
Module for inferencing.
"""

# Import Libraries
import numpy as np
import torch
import torch.nn.functional as F
from utils import dice_coeff, multiclass_dice_coeff
from skimage.metrics import hausdorff_distance
from scipy.spatial.distance import directed_hausdorff

# Function to evaluate Dice score for each class in the test set
def evaluate_dice_score(model, dataloader, num_classes=4, multi_class=True, device='cpu', threshold=0.5):
    model.eval()  # Set model to evaluation mode
    dice_scores = [0.0] * num_classes  # List to store Dice score for each class
    # count = 0  # Number of samples
    batch_count = len(dataloader) # Number of batches

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            # Extract data from the sample
            images, true_masks = sample['image'], sample['masks_all']
            images = images.permute(0, 3, 1, 2)

            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            true_masks = F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float()

            outputs = model(images)  # Shape: (batch_size, 4, height, width)
            predictions = F.one_hot(outputs.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()

            # Calculate Dice score for each class
            for i in range(num_classes):  # Loop over each class
                # dice_scores[i] += dice_coeff(
                #     predictions[:, i, :, :].flatten(0, 1),
                #     true_masks[:, i, :, :].flatten(0, 1),
                #     reduce_batch_first=False
                #     )

                dice_scores[i] += multiclass_dice_coeff(
                    predictions[:, i, :, :],
                    true_masks[:, i, :, :],
                    reduce_batch_first=False
                )

            # count += images.size(0)  # Update sample count
            # batch_count += 1 # Update batch count

    # Average Dice scores across the dataset
    # avg_dice_scores = [score / count for score in dice_scores]
    avg_dice_scores_batch = [score / batch_count for score in dice_scores]
    return avg_dice_scores_batch


class SegmentationMetrics:
    def __init__(self, prediction, ground_truth, num_classes=4, device="cpu"):
        """
        :param prediction: (N, H, W) torch tensor with class labels
        :param ground_truth: (N, H, W) torch tensor with class labels
        :param num_classes: number of classes including background (default 4)
        :param device: "cpu" or "cuda"
        """
        self.pred = prediction.to(device)
        self.gt = ground_truth.to(device)
        self.num_classes = num_classes
        self.device = device

    def _binary_metrics(self, cls):
        pred_bin = (self.pred == cls).float()
        gt_bin = (self.gt == cls).float()

        dims = (1, 2)  # H, W
        intersection = (pred_bin * gt_bin).sum(dim=dims)
        union = ((pred_bin + gt_bin) > 0).float().sum(dim=dims)
        sum_pred_gt = pred_bin.sum(dim=dims) + gt_bin.sum(dim=dims)

        dice = (2. * intersection + 1e-6) / (sum_pred_gt + 1e-6)
        jaccard = (intersection + 1e-6) / (union + 1e-6)

        # Hausdorff Distance (still using CPU/scipy for now)
        hausdorff_list = []
        for i in range(pred_bin.shape[0]):
            pb = pred_bin[i].cpu().numpy()
            gb = gt_bin[i].cpu().numpy()
            if pb.any() and gb.any():
                pb_pts = torch.nonzero(torch.tensor(pb)).numpy()
                gb_pts = torch.nonzero(torch.tensor(gb)).numpy()
                hd = max(
                    directed_hausdorff(pb_pts, gb_pts)[0],
                    directed_hausdorff(gb_pts, pb_pts)[0]
                )
            else:
                hd = 0.0
            hausdorff_list.append(hd)

        return dice.mean().item(), jaccard.mean().item(), sum(hausdorff_list) / len(hausdorff_list)

    def evaluate_per_class(self, class_names=None):
        results = {}
        for cls in range(1, self.num_classes):  # Skip background
            name = class_names[cls] if class_names else f"Class {cls}"
            dice, jaccard, hausdorff = self._binary_metrics(cls)
            results[name] = {
                "Dice Score": dice,
                "Jaccard Index (IoU)": jaccard,
                "Hausdorff Distance": hausdorff
            }
        return results
