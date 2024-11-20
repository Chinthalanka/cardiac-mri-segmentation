import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Convert target to one-hot encoding
    # target = torch.nn.functional.one_hot(target.long(), num_classes=input.shape[1]).permute(0, 3, 1, 2).float()

    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def weighted_dice_loss(pred, target, class_weights, smooth = 1e-6):
    # pred and target are of size (batch_size, num_classes, height, width)
    # class_weights is of size (num_classes,)

    batch_size, num_classes, height, width = pred.size()

    # Flatten the tensors to compute Dice score
    pred = pred.view(batch_size, num_classes, -1)
    target = target.view(batch_size, num_classes, -1)

    # Compute intersection and union for each class
    intersection = torch.sum(pred * target, dim=2)
    union = torch.sum(pred, dim=2) + torch.sum(target, dim=2)

    # Compute Dice coefficient for each class
    dice_score = (2.0 * intersection + smooth) / (union + smooth)

    # Convert dice score to dice loss
    dice_loss = 1.0 - dice_score

    # Multiply by class weights and average over batch
    weighted_dice_loss = torch.mean(class_weights * dice_loss)

    return weighted_dice_loss
