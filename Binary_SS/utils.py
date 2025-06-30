import torch
import torchvision
from dataset import CUB200
from torch.utils.data import DataLoader
import logging

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

def get_loaders(
    dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CUB200(
        data_set_root=dir, image_size=128,
                     transform=train_transform, test_train=0, return_masks=True
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CUB200(
        data_set_root=dir, image_size=128,
                     transform=val_transform, test_train=1, return_masks=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def compute_accuracy(preds, targets):
    num_correct = (preds == targets).sum()
    num_pixels = torch.numel(preds)
    return num_correct, num_pixels

def compute_dice_score(preds, targets, eps=1e-8):
    intersection = (preds * targets).sum()
    total = preds.sum() + targets.sum()
    return (2 * intersection) / (total + eps)

def compute_iou(preds, targets, eps=1e-8):
    intersection = (preds * targets).sum()
    union = ((preds + targets) > 0).float().sum()
    return intersection / (union + eps)

def evaluate(loader, model, curr_epoch, device="cuda", logger=None):
    num_correct = 0
    num_pixels = 0
    total_dice = 0
    total_iou = 0
    model.eval()

    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            correct, pixels = compute_accuracy(preds, y)
            dice = compute_dice_score(preds, y)
            iou = compute_iou(preds, y)

            num_correct += correct
            num_pixels += pixels
            total_dice += dice
            total_iou += iou

    acc = num_correct / num_pixels * 100
    mean_dice = total_dice / len(loader)
    mean_iou = total_iou / len(loader)

    msg = f"Epoch {curr_epoch}\nAccuracy: {acc:.2f}%, Dice: {mean_dice:.4f}, IoU: {mean_iou:.4f}"
    
    if logger:
        logger.info(msg)
    else:
        print(msg)

    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y, _, _) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.float().unsqueeze(1), f"{folder}{idx}.png")

    model.train()