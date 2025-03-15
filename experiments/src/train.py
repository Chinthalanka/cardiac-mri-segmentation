"""
Module to train a Pytorch model
"""

# Import Libraries
import mlflow
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from utils import dice_loss, weighted_dice_loss, evaluate, EarlyStopping

# Setup logger
logger = logging.getLogger('__name__')
logger.setLevel(logging.INFO)


def train_model(
        dataset,
        model,
        training_dataset,
        validation_dataset,
        train_dataloader,
        val_dataloader,
        device,
        model_name: str = None,
        epochs: int = 5,
        start_epoch: int = 1,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-5,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        exp_run_id: str = None,
        multi_class: bool = False,
        class_weights=None,
        dir_checkpoint=None
):

    if class_weights is None:
        class_weights = [0.3, 0.4, 0.3]
    class_weights = torch.tensor(class_weights).to(device=device)

    n_train = len(training_dataset)
    n_val = len(validation_dataset)

    # Start an MLflow run
    with mlflow.start_run() as run:
        # Get the run_id of the current active run
        run_id = run.info.run_id

        # Set early stopping parameters
        patience = 10  # Number of epochs to wait for improvement
        min_delta = 0.001  # Minimum change to count as an improvement
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        logger.info(f'''Starting training:
            Dataset:         {dataset}
            Model:           {model_name}
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Images scaling:  {img_scale}
            Mixed Precision: {amp}
        ''')

        params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'training_set_size': n_train,
            'validation_set_size': n_val,
            'img_scale': img_scale,
            'mixed_precision': amp
        }

        mlflow.log_params(params)
        mlflow.set_tag("dataset", dataset)
        mlflow.set_tag("model", model_name)

        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir=f'./logs/tensorboard_runs/{exp_run_id}')

        # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=True)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        criterion = nn.CrossEntropyLoss() if multi_class else nn.BCEWithLogitsLoss()
        criterion.to(device=device)
        global_step = 0
        best_val_loss = np.inf

        # Begin training
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            epoch_loss = 0
            epoch_dice_loss = 0
            # running_val_dice_loss = 0
            # running_val_dice_loss_steps = 0

            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                for batch in train_dataloader:
                    images, masks = batch['image'], batch['masks_all']
                    images = images.permute(0, 3, 1, 2)
                    # logger.info(f"images shape: {images.shape}")
                    # logger.info(f"true masks shape: {masks.shape}")

                    assert images.shape[1] == model.n_channels, \
                        f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = masks.to(device=device, dtype=torch.long)

                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                        masks_pred = model(images)
                        # logger.info(f"masks pred shape: {masks_pred.shape}")
                        if model.n_classes == 1:
                            loss_ce = criterion(masks_pred.squeeze(1), true_masks.float())
                            loss_dice = dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(),
                                                  multiclass=False)
                            loss = 0.5 * loss_ce + 0.5 * loss_dice
                        else:
                            loss_ce = criterion(masks_pred, true_masks)  # true_masks.squeeze(1).long()

                            if multi_class:
                                # Ignore background when computing Dice loss
                                loss_dice = dice_loss(
                                    F.softmax(masks_pred, dim=1).float()[:, 1:],
                                    F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float()[:, 1:],
                                    multiclass=True
                                )
                            else:
                                loss_dice = weighted_dice_loss(
                                    pred=F.sigmoid(masks_pred > 0.5).float(),
                                    target=true_masks.float(),
                                    class_weights=class_weights
                                )

                            loss = 0.3 * loss_ce + 0.7 * loss_dice

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    epoch_dice_loss += loss_dice.item()
                    # pbar.set_postfix(**{'loss (batch)': loss.item()})
                    pbar.set_postfix(**{'total_loss (batch)': loss.item(), 'dice_loss (batch)': loss_dice.item(),
                                        'ce_loss (batch)': loss_ce.item()})

                    # Evaluation round
                    division_step = (n_train // (5 * batch_size))
                    if division_step > 0:
                        if global_step % division_step == 0:
                            val_score = evaluate(model, val_dataloader, device, amp, multi_class=multi_class)
                            # running_val_dice_loss += 1 - val_score
                            # running_val_dice_loss_steps += 1
                            scheduler.step(val_score)
                            logger.info(f'Validation Dice score: {val_score:.4f}')

            # Compute average losses
            avg_loss = epoch_loss / len(train_dataloader)
            avg_dice_loss = epoch_dice_loss / len(train_dataloader)
            # avg_dice_val_loss = running_val_dice_loss / running_val_dice_loss_steps
            avg_dice_val_loss = 1 - evaluate(model, val_dataloader, device, amp, multi_class=multi_class)
            logger.info(f"Validation Dice Loss at Epoch {epoch}: {avg_dice_val_loss:.4f}")

            # Save checkpoints of best performing models
            if save_checkpoint and (avg_dice_val_loss < best_val_loss):
                best_val_loss = avg_dice_val_loss
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                # state_dict['mask_values'] = training_dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logger.info(f'Checkpoint {epoch} saved! --> best_val_loss: {best_val_loss:.4f}')

            # Log metrics to TensorBoard
            writer.add_scalars("dice_loss_and_score",
                               {
                                   'training_loss': avg_dice_loss,
                                   'validation_loss': avg_dice_val_loss,
                                   'training_score': (1 - avg_dice_loss),
                                   'validation_score': (1 - avg_dice_val_loss)
                               },
                               epoch)
            writer.add_scalar("hybrid_loss_training", avg_loss, epoch)
            writer.add_scalar("dice_loss/training", avg_dice_loss, epoch)
            writer.add_scalar("dice_loss/validation", avg_dice_val_loss, epoch)
            writer.add_scalar("dice_score/training", (1 - avg_dice_loss), epoch)
            writer.add_scalar("dice_score/validation", (1 - avg_dice_val_loss), epoch)

            # Check for early stopping
            early_stopping(avg_dice_val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                logger.info(f'Early stopping triggered at epoch {epoch + 1}')
                break

        # Flush and close the writer when done
        writer.flush()
        writer.close()

    return run_id


# Weight Initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)