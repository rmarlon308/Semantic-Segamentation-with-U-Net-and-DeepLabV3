# Image Segmentation with U-Net and DeepLabV3 on CUB-200-2011 and Pascal VOC2012

This repository implements a semantic segmentation pipeline using two popular deep learning architectures: **U-Net** and **DeepLabV3**. The models are trained and evaluated on two datasets:
- **Pascal VOC2012** for multiclass image segmentation
- **CUB-200-2011** for bird image segmentation

## Bird Image Segmentation with U-Net over CUB-200-2011

### Dataset

The [CUB-200-2011](https://www.kaggle.com/datasets/wenewone/cub2002011) dataset contains:
- **11,788 images** of **200 bird species**
- Ground-truth **bounding boxes**, **part locations**, and **segmentation masks**

In this project, we focus on the **segmentation masks** provided for each image.

### Model

We use the U-Net architecture for pixel-wise binary segmentation. The network consists of:
- **Encoder**: series of convolutional blocks that downsample the image, capturing **contextual and semantic information** about the scene.
- **Decoder**: upsampling blocks with skip connections from the encoder, enabling **precise localization** and reconstruction of object boundaries.
- **Output**: a binary mask representing the segmented bird region

Model is implemented in **PyTorch** and trained from scratch.

### Results

After 100 epochs of training, the model achieved the following performance on the validation set:

| Metric       | Value     |
|--------------|-----------|
| Accuracy     | 97.63%    |
| Dice Score   | 0.9352    |
| IoU          | 0.8787    |

#### Example Output

Below is a comparison between the input image, the ground truth mask, and the predicted mask:

<p align="center">
  <img src="assets/example_output.svg" alt="Segmentation Results" width="600"/>
</p>

## Multiclass Image Segmentation with U-Net and DeepLabV3 on Pascal VOC2012

This project focuses on pixel-wise multiclass semantic segmentation using **U-Net** and **DeepLabV3** architectures trained on the **Pascal VOC 2012** dataset. The goal is to classify each pixel into one of 21 categories (20 object classes + background).

---

### Dataset: Pascal VOC 2012

The [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset includes:
- 1,464 training images
- 1,449 validation images
- Pixel-level segmentation masks for 21 semantic classes

To enhance generalization, we apply custom **data augmentations** using the **Albumentations** library, including:
- Horizontal flips
- Color jitter
- Normalization

---

### Models

We implement and evaluate two popular semantic segmentation models:

- **U-Net**
- **DeepLabV3**

Both architectures use a **ResNet-34** encoder **pretrained on ImageNet**, providing strong feature representations even with limited data.

---

### DeepLabV3 Architecture

DeepLabV3 is a state-of-the-art semantic segmentation model designed to handle complex object shapes and scales.

- **Encoder**: A convolutional backbone (ResNet-34) extracts deep semantic features using **dilated (atrous) convolutions** to preserve resolution while expanding the receptive field.
- **ASPP (Atrous Spatial Pyramid Pooling)**: Captures **multi-scale context** using parallel convolutions with different dilation rates.
- **Output**: Generates a dense per-pixel class probability map.

> Unlike U-Net, DeepLabV3 does not use an explicit decoder. It leverages ASPP for multi-scale understanding before upsampling to the original resolution.

---

### Improvement Strategies

To boost segmentation performance, we explored the following strategies:

- Use **class weights** in the Cross-Entropy (CE) loss to mitigate class imbalance.
- Combine **Cross-Entropy + Dice Loss** to better handle small or thin objects.
- Expand the training set to **85% of total images** (instead of the default ~50%).
- Replace U-Net with **DeepLabV3**, which provides stronger context modeling.
- Add a **learning rate scheduler** starting at epoch 80 to refine training.

---

### Experimental Results

We trained multiple configurations for 50 epochs to evaluate the impact of each strategy:

#### Results (50% training data)

| Architecture | Backbone  | LR    | Improvement Strategy                        | Mean IoU |
|--------------|-----------|-------|---------------------------------------------|----------|
| U-Net        | ResNet-34 | 1e-4  | CE Loss + Class Weights (CW)                | 0.3610   |
| U-Net        | ResNet-34 | 1e-4  | CE + CW + Dice Loss                         | 0.3945   |
| U-Net        | ResNet-34 | 2e-4  | CE + CW + Dice Loss                         | 0.4512   |
| DeepLabV3    | ResNet-34 | 2e-4  | CE + CW + Dice Loss                         | 0.4838   |

#### Results (85% training data)

| Architecture | Backbone  | LR    | Improvement Strategy                        | Mean IoU |
|--------------|-----------|-------|---------------------------------------------|----------|
| U-Net        | ResNet-34 | 1e-4  | CE(CW) + Dice + 85% Training Data           | 0.5237   |
| U-Net        | ResNet-34 | 2e-4  | CE(CW) + Dice + 85% Training Data           | 0.5624   |
| DeepLabV3    | ResNet-34 | 2e-4  | CE(CW) + Dice + 85% Training Data           | 0.5699   |

Based on these results, we selected the best-performing configuration (DeepLabV3 + all improvements) for further training up to 100 epochs. Additionally, we introduced a **StepLR scheduler starting at epoch 30** with steps of 15 epochs:

| Architecture | Backbone  | LR    | Strategy                                             | Mean IoU |
|--------------|-----------|-------|------------------------------------------------------|----------|
| DeepLabV3    | ResNet-34 | 2e-4  | CE(CW) + Dice + 85% Training + StepLR (start @ 40)   | 0.5726   |
| DeepLabV3    | ResNet-34 | 2e-4  | CE(CW) + Dice + 85% Training Data                    | 0.5877   | 

---

#### Example Output

Below is a comparison between the input image, the ground truth segmentation mask, and the predicted mask using the best configuration:

<p align="center">
  <img src="assets/VOC_predictions.svg" alt="Segmentation Results VOC" width="600"/>
</p>

---
