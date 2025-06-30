import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
import torch.optim as optim

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    evaluate,
    save_predictions_as_imgs,
    get_logger,
)

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 101
NUM_WORKERS = 2
IMAGE_SIZE = 128
PIN_MEMORY = True
LOAD_MODEL = False
DATA_PATH = "./data/"
SAVE_AFTER = 10


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_index, (data, targets, bbox, label) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)

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


def main():
    image_size = 128
    train_transform = A.Compose([A.SmallestMaxSize(max_size=image_size),
                             A.RandomCrop(height=image_size, width=image_size),
                             A.HorizontalFlip(p=0.5),
                             A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                             A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                             A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                             A.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                            ToTensorV2()], 
                            bbox_params=A.BboxParams(format='coco',
                                                     min_area=0, min_visibility=0.0, 
                                                     label_fields=['class_labels']))

    transform = A.Compose([A.SmallestMaxSize(max_size=image_size),
                        A.CenterCrop(height=image_size, width=image_size),
                        A.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                        ToTensorV2()], 
                        bbox_params=A.BboxParams(format='coco',
                                                min_area=0, min_visibility=0.0, 
                                                label_fields=['class_labels']))
    
    logger = get_logger("segmentation_metrics.log")
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_loder, val_loader = get_loaders(DATA_PATH, BATCH_SIZE, train_transform, transform, NUM_WORKERS, PIN_MEMORY)
    scaler = torch.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loder, model, optimizer, loss_fn, scaler)


        if epoch%SAVE_AFTER == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

            evaluate(val_loader, model, epoch, device=DEVICE, logger=logger)

            save_predictions_as_imgs(
                val_loader, model, folder="saved_images/", device=DEVICE
            )


if __name__ == '__main__':
    main()



