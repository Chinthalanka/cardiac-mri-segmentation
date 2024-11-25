"""
This module contains evaluation metrics related to ROI predictor model.
"""

# Import libraries
import torch

def calculate_iou(pred_box, gt_box):
    """
    Calculate Intersection over Union (IoU) for a pair of bounding boxes.
    Args:
        pred_box: Tensor [x_min, y_min, x_max, y_max], predicted bounding box.
        gt_box: Tensor [x_min, y_min, x_max, y_max], ground truth bounding box.
    Returns:
        IoU: Float, Intersection over Union.
    """
    # Intersection coordinates
    x_min = max(pred_box[0], gt_box[0])
    y_min = max(pred_box[1], gt_box[1])
    x_max = min(pred_box[2], gt_box[2])
    y_max = min(pred_box[3], gt_box[3])

    # Intersection area
    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)

    # Union area
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = pred_area + gt_area - intersection

    # IoU
    return intersection / union if union > 0 else 0.0


def evaluate_iou(dataloader, model, device):
    """
    Evaluate IoU for a dataset using a trained model.
    Args:
        dataloader: DataLoader for evaluation.
        model: Trained ROI predictor.
        device: Torch device (CPU or CUDA).
    Returns:
        avg_iou: Average IoU across the dataset.
    """
    model.eval()
    iou_scores = []

    with torch.no_grad():
        for batch in dataloader:
            images, gt_boxes = batch['org_image'], batch['bounding_box']
            images = images.permute(0, 3, 1, 2)

            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            gt_boxes = gt_boxes.to(device=device, dtype=torch.float32)

            pred_boxes = model(images)  # Predicted bounding boxes

            for pred_box, gt_box in zip(pred_boxes, gt_boxes):
                iou = calculate_iou(pred_box.cuda(), gt_box.cuda())  # pred_box.cpu(), gt_box.cpu()
                iou_scores.append(iou)
    model.train()

    return sum(iou_scores) / len(iou_scores)


def evaluate_regression_metrics(dataloader, model, device):
    """
    Evaluate regression metrics (MSE and MAE) for bounding box prediction.
    Args:
        dataloader: DataLoader for evaluation.
        model: Trained ROI predictor.
        device: Torch device (CPU or CUDA).
    Returns:
        mse: Mean Squared Error across the dataset.
        mae: Mean Absolute Error across the dataset.
    """
    model.eval()
    total_mse, total_mae, n_samples = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in dataloader:
            images, gt_boxes = batch['org_image'], batch['bounding_box']
            images = images.permute(0, 3, 1, 2)

            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            gt_boxes = gt_boxes.to(device=device, dtype=torch.float32)

            pred_boxes = model(images)  # Predicted bounding boxes

            mse = ((pred_boxes - gt_boxes) ** 2).sum(dim=1).mean().item()
            mae = (pred_boxes - gt_boxes).abs().sum(dim=1).mean().item()

            total_mse += mse * len(images)
            total_mae += mae * len(images)
            n_samples += len(images)
    model.train()

    return total_mse / n_samples, total_mae / n_samples

