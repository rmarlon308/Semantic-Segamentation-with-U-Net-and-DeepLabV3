import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim
import segmentation_models_pytorch as smp
from collections import Counter
from tqdm import tqdm
import numpy as np

from utils import (
    load_checkpoint,
    save_checkpoint,
    evaluate,
    save_predictions_as_imgs,
    get_logger,
    get_loadersVOC,
)

LEARNING_RATE = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 101
NUM_WORKERS = 4
IMAGE_SIZE = 256
PIN_MEMORY = True
LOAD_MODEL = False
DATA_PATH = "./data/"
SAFE_AFTER = 10
CUSTOM_SPLIT = True
TRAIN_SPLIT_SIZE = 0.85
START_DECAY_EPOCH = 30
STEP_SIZE = 15 # In Epoch 45 is the first decay

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, ce_weight=1.0, class_weights=None):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, preds, targets):
        return self.dice_weight * self.dice(preds, targets) + \
               self.ce_weight * self.ce(preds, targets)


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_index, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward
        with torch.amp.autocast(device_type=DEVICE):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())


def compute_class_frequencies(loader, num_classes=21):
    class_counts = Counter()
    total_pixels = 0

    for _, masks in tqdm(loader):
        masks = masks.view(-1)
        masks = masks[masks != 255]  # ignore black pixels
        classes, counts = torch.unique(masks, return_counts=True)

        for cls, count in zip(classes.tolist(), counts.tolist()):
            class_counts[cls] += count
            total_pixels += count

    return class_counts, total_pixels

def compute_class_weights(class_counts, total_pixels, num_classes=21):
    weights = torch.ones(num_classes)
    for cls in range(num_classes):
        if cls in class_counts:
            weights[cls] = total_pixels / (num_classes * class_counts[cls])
        else:
            weights[cls] = 0.0  # If the class never appears

    # Weights normalization
    weights = weights / weights.mean()
    return weights


def main():

    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    logger = get_logger("segmentation_metricsVOC.log")

    # model = smp.Unet(
    #     encoder_name="resnet34",       
    #     encoder_weights="imagenet",
    #     in_channels=3,
    #     classes=21 
    # ).to(DEVICE)

    model = smp.DeepLabV3(
        encoder_name="resnet34",       
        encoder_weights="imagenet",
        in_channels=3,
        classes=21 
    ).to(DEVICE) # UNet or DeepLabV3 model with Pre-trained resnet34 as encoder

    train_loader, val_loader = get_loadersVOC(
        DATA_PATH, BATCH_SIZE, train_transform, val_transform, NUM_WORKERS, PIN_MEMORY,
        CUSTOM_SPLIT, TRAIN_SPLIT_SIZE
    )

    counts, total = compute_class_frequencies(train_loader, num_classes=21)
    class_weights = compute_class_weights(counts, total, num_classes=21)
    print("Class weights:", class_weights)

    print('Pre-trained Model Evaluation:')
    evaluate(val_loader, model, -1, device=DEVICE, logger=logger)

    loss_fn = CombinedLoss(class_weights=class_weights.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, 
    #     step_size=STEP_SIZE, 
    #     gamma=0.1
    # )

    scaler = torch.amp.GradScaler()
    
    # prev_lr = optimizer.param_groups[0]["lr"]  # LR inicial

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # if epoch >= START_DECAY_EPOCH:
        #     scheduler.step()
        #     current_lr = optimizer.param_groups[0]["lr"]
        #     if current_lr != prev_lr:
        #         print(f"Epoch {epoch}: Learning rate decayed to {current_lr:.2e}")
        #     prev_lr = current_lr

        if epoch%SAFE_AFTER == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

            evaluate(val_loader, model, epoch, device=DEVICE, logger=logger)

            save_predictions_as_imgs(
                val_loader, model, folder="saved_images/", device=DEVICE
            )

    # last_checkpoint = torch.load("my_checkpoint.pth.tar")
    # load_checkpoint(last_checkpoint, model)
    # evaluate(val_loader, model, 101, device=DEVICE, logger=logger)
    # save_predictions_as_imgs(
    #             val_loader, model, folder="saved_images/", device=DEVICE
    #         )


if __name__ == '__main__':
    main()



