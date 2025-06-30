import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import os
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

VOC_COLORMAP = [
    (0, 0, 0),        # background
    (128, 0, 0),      # aeroplane
    (0, 128, 0),      # bicycle
    (128, 128, 0),    # bird
    (0, 0, 128),      # boat
    (128, 0, 128),    # bottle
    (0, 128, 128),    # bus
    (128, 128, 128),  # car
    (64, 0, 0),       # cat
    (192, 0, 0),      # chair
    (64, 128, 0),     # cow
    (192, 128, 0),    # diningtable
    (64, 0, 128),     # dog
    (192, 0, 128),    # horse
    (64, 128, 128),   # motorbike
    (192, 128, 128),  # person
    (0, 64, 0),       # pottedplant
    (128, 64, 0),     # sheep
    (0, 192, 0),      # sofa
    (128, 192, 0),    # train
    (0, 64, 128)      # tvmonitor
]


class VOC(Dataset):
    def __init__(self, 
                 data_path, 
                 train_transform,
                 val_transform, 
                 train_test,
                 custom_split = False, 
                 train_size = 0.85):
        super(VOC, self).__init__()

        self.image_root_dir = os.path.join(data_path, "VOC2012/JPEGImages")
        self.mask_root_dir = os.path.join(data_path, "VOC2012/SegmentationClass")

        if custom_split:
            data = np.array(self.get_file_list(os.path.join(data_path, "VOC2012/ImageSets/Segmentation/trainval.txt")))
            np.random.seed(42)
            indices = np.random.permutation(len(data))
            split_idx = int(len(data) * train_size)
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            self.train_split = data[train_indices]
            self.val_split = data[val_indices]
        else:
            data = np.array(self.get_file_list(os.path.join(data_path, "VOC2012/ImageSets/Segmentation/trainval.txt")))
            self.train_split = self.get_file_list(os.path.join(data_path, "VOC2012/ImageSets/Segmentation/train.txt"))
            self.val_split = self.get_file_list(os.path.join(data_path, "VOC2012/ImageSets/Segmentation/val.txt"))

        self.train_test = train_test
        self.transform = train_transform  if self.train_test == 0 else val_transform

    def get_file_list(self, file):
        with open(file, 'r') as f:
            lines = [line.strip() for line in f]
        return lines

    def rgb_to_class(self, mask):
        """
        Convierte una máscara RGB VOC a índices de clase entre 0 y 20.
        mask: np.array HxWx3 (uint8)
        returns: np.array HxW con valores entre 0 y 20
        """
        h, w, _ = mask.shape
        class_mask = np.zeros((h, w), dtype=np.uint8)

        for idx, color in enumerate(VOC_COLORMAP):
            matches = np.all(mask == color, axis=-1)
            class_mask[matches] = idx

        return class_mask
    
    def getSplitsSize(self):
        return [len(self.train_split), len(self.val_split), len(self.train_split) + len(self.val_split)]

    def __len__(self):
        if self.train_test == 0:
            return len(self.train_split)
        else:
            return len(self.val_split)

    def __getitem__(self, index):
        split = self.train_split if self.train_test == 0 else self.val_split
        item = split[index]

        img_path = os.path.join(self.image_root_dir, item + '.jpg')
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.mask_root_dir, item + '.png')
        mask_rgb = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
        mask_class = self.rgb_to_class(mask_rgb).astype("uint8")


        if self.transform:
            augmented = self.transform(image=image, mask=mask_class)
            image = augmented["image"]
            mask = augmented["mask"].long()

        return image, mask



def test2():
    data_path = './data/'
    image_size = 128
    test_transform = A.Compose([
        A.Resize(128, 128),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_transform = A.Compose([A.Resize(140, 140),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()])
    set = VOC(data_path, train_transform, test_transform, 0, True, 0.85)
    print(set.__len__())
    train_dl = DataLoader(
        set,
        64,
        True,
        num_workers=4
    )

    print(len(train_dl))


    set.update_set(1)
    print(set.__len__())

    test_dl = DataLoader(
        set,
        64,
        False,
        num_workers=4
    )

    print(len(test_dl))

    

    

def test():
    data_path = './data/'
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
    return_masks = True
    train_dataset = CUB200(data_set_root=data_path, image_size=image_size,
                     transform=train_transform, test_train=0, return_masks=return_masks)
    test_dataset = CUB200(data_set_root=data_path, image_size=image_size,
                     transform=train_transform, test_train=1, return_masks=return_masks)

    print(train_dataset.__len__())
    print(test_dataset.__len__())

if __name__ == "__main__":
    test2()



