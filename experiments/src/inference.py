"""
Module for inferencing.
"""

# Import Libraries
import torch
import torch.nn.functional as F
from utils import dice_coeff, multiclass_dice_coeff

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