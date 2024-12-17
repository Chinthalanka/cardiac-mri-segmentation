"""
Module to visualize predictions
"""

# Import Libraries
import torch
import matplotlib.pyplot as plt


def visualize_predictions(dataloader, model, device, num_samples=5, classes=None):
    """
    Visualizes predictions alongside the original image and ground truth mask.

    Parameters:
        dataloader (torch.utils.data.DataLoader): The DataLoader providing the data.
        model (torch.nn.Module): The trained model for predictions.
        device (torch.device): The device to run the model (e.g., 'cuda' or 'cpu').
        num_samples (int): Number of samples to visualize from the dataloader.
        classes (list): Optional list of class names for displaying masks.
    """
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for i, sample in enumerate(dataloader):
            if i >= num_samples:
                break

            images, masks = sample['image'], sample['masks_all']
            images = images.permute(0, 3, 1, 2)

            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks = masks.to(device=device, dtype=torch.long)

            # Get model predictions
            predictions = model(images)  # Shape: [batch_size, num_classes, height, width]
            predictions = torch.argmax(predictions, dim=1)  # Shape: [batch_size, height, width]

            batch_size = images.size(0)
            for j in range(batch_size):
                if num_samples <= 0:
                    break

                num_samples -= 1

                # Move tensors to CPU for visualization
                image = images[j].cpu().permute(1, 2, 0).numpy()  # Convert to HWC format
                ground_truth = masks[j].cpu().numpy()  # Ground truth mask
                prediction = predictions[j].cpu().numpy()  # Predicted mask

                # Plot the results
                fig, ax = plt.subplots(1, 3, figsize=(8, 5))

                ax[0].imshow(image)
                ax[0].set_title("Original Image")
                ax[0].axis("off")

                ax[1].imshow(ground_truth, cmap="viridis", interpolation="none")
                ax[1].set_title("Ground Truth Mask")
                if classes:
                    ax[1].set_xticks([])
                    ax[1].set_yticks([])

                ax[2].imshow(prediction, cmap="viridis", interpolation="none")
                ax[2].set_title("Predicted Mask")
                if classes:
                    ax[2].set_xticks([])
                    ax[2].set_yticks([])

                plt.tight_layout()
                plt.show()