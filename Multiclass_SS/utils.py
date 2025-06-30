import torch
import torchvision
from dataset import CUB200, VOC
from torch.utils.data import DataLoader
import logging

import os
import numpy as np
from PIL import Image
from collections import Counter
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import make_grid

VOC_COLORMAP = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
    (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
    (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
    (0, 64, 128)
]

LEARNING_RATE = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 101
NUM_WORKERS = 4
IMAGE_SIZE = 128
PIN_MEMORY = True
LOAD_MODEL = False
DATA_PATH = "./data/"
SAFE_AFTER = 10

def get_logger(log_path="metrics_log.txt"):
    logger = logging.getLogger("segmentation_logger")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loadersVOC(
    dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    custom_split = False,
    train_size = 0.85
):
    train_dataset = VOC(dir, train_transform, val_transform, train_test=0, custom_split=custom_split, train_size=train_size)

    train_loader = DataLoader(
        train_dataset, # Train
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_dataset = VOC(dir, train_transform, val_transform, train_test=1, custom_split=custom_split, train_size=train_size)

    val_loader = DataLoader(
        val_dataset, #Test
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    assert train_dataset.getSplitsSize() == val_dataset.getSplitsSize()
    train_size, val_size, total = val_dataset.getSplitsSize()
    print("Train size:", train_size, " ", train_size/total)
    print("Val size:", val_size, " ", val_size/total)
    
    return train_loader, val_loader

def compute_dice_per_class(preds, targets, num_classes, ignore_index=255, eps=1e-8):
    dices = []
    for cls in range(num_classes):
        if cls == ignore_index:
            dices.append(np.nan)
            continue

        pred_inds = (preds == cls)
        target_inds = (targets == cls)

        if torch.sum(target_inds) == 0 and torch.sum(pred_inds) == 0:
            dices.append(np.nan)
            continue

        intersection = (pred_inds & target_inds).sum().item()
        total = pred_inds.sum().item() + target_inds.sum().item()
        dice = 2 * intersection / (total + eps)
        dices.append(dice)
    return dices

def compute_accuracy(preds, targets):
    return (preds == targets).sum(), targets.numel()


def compute_iou_per_class(preds, targets, num_classes=21, ignore_index=255):
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)

        valid_mask = (targets != ignore_index)
        pred_inds = pred_inds & valid_mask
        target_inds = target_inds & valid_mask

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)

    return ious

def evaluate(loader, model, curr_epoch, device="cuda", logger=None):
    num_correct = 0
    num_pixels = 0
    all_ious = []
    all_dices = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device) 

            logits = model(x)                  
            preds = torch.argmax(logits, dim=1) 

            correct, pixels = compute_accuracy(preds, y)
            ious = compute_iou_per_class(preds, y, num_classes=21, ignore_index=255)
            dices = compute_dice_per_class(preds, y, num_classes=21, ignore_index=255)

            num_correct += correct
            num_pixels += pixels
            all_ious.append(ious)
            all_dices.append(dices)

    acc = num_correct / num_pixels * 100

    all_ious = np.array(all_ious)
    all_dices = np.array(all_dices)

    mean_ious_per_class = np.nanmean(all_ious, axis=0)
    mean_dices_per_class = np.nanmean(all_dices, axis=0)

    mean_iou = np.nanmean(mean_ious_per_class)
    mean_dice = np.nanmean(mean_dices_per_class)

    msg = f"Epoch {curr_epoch}\nAccuracy: {acc:.2f}%, Mean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    model.train()

def decode_segmap(mask):
    mask = mask.cpu().numpy()
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(VOC_COLORMAP):
        color_mask[mask == cls_id] = color
    return Image.fromarray(color_mask)

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda", nrow = 8):
    os.makedirs(folder, exist_ok=True)
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            logits = model(x)                    
            preds = torch.argmax(logits, dim=1)
            # print("Unique pred classes:", torch.unique(preds))

        batch_input = []
        batch_pred = []
        batch_gt = []

        for i in range(x.shape[0]):
            input_img = x[i].cpu()
            pred_mask = decode_segmap(preds[i]) 
            gt_mask = decode_segmap(y[i])     

            pred_tensor = torchvision.transforms.ToTensor()(pred_mask)
            gt_tensor = torchvision.transforms.ToTensor()(gt_mask)

            batch_input.append(input_img)
            batch_pred.append(pred_tensor)
            batch_gt.append(gt_tensor)


        input_grid = make_grid(torch.stack(batch_input), nrow=nrow, normalize=True)
        pred_grid = make_grid(torch.stack(batch_pred), nrow=nrow)
        gt_grid = make_grid(torch.stack(batch_gt), nrow=nrow)

        torchvision.utils.save_image(input_grid, os.path.join(folder, f"inputs_batch_{idx}.png"))
        torchvision.utils.save_image(gt_grid, os.path.join(folder, f"gts_batch_{idx}.png"))
        torchvision.utils.save_image(pred_grid, os.path.join(folder, f"preds_batch_{idx}.png"))

    model.train()

def main():

    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_loder, val_loader = get_loadersVOC(
        DATA_PATH, BATCH_SIZE, train_transform, 
        transform, NUM_WORKERS, PIN_MEMORY, 
        custom_split = False,
        train_size = 0.85
    )

if __name__ == '__main__':
    main()