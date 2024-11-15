---
layout: post
title: Fine-Tuning Vision Transformer (ViT) on Tiny ImageNet Dataset
date: 2024-11-15
description: In this post, I'll generally introduce how to fine-tune a ViT model on a tiny ImageNet dataset.
tags: fine-tune
categories: LLM
related_posts: false
---


## Introduction

This document provides a detailed overview of the strategy employed to fine-tune a Vision Transformer (ViT) on the Tiny ImageNet dataset, achieving a validation accuracy of **90.5% within 10 epochs**.

## Dataset Description

- **Dataset**: Tiny ImageNet
- **Number of Classes**: 200
- **Image Size**: 64x64 resized to 384x384 for ViT

## Model Configuration

- **Model**: ViT-Base with patch size 16 (`vit_base_patch16_384`)
- **Pretrained Weights**: Used pretrained weights from ImageNet
- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate**: 1e-4
- **Weight Decay**: 0.01
- **Scheduler**: Cosine Annealing Learning Rate
- **Loss Function**: Soft Target Cross-Entropy (for Mixup/CutMix)
- **Augmentation**: RandAugment, Random Erasing, Mixup, and CutMix

## Strategy

### Data Preprocessing
1. **Image Resizing**: 
   - Images were resized to 384x384 to match the input dimensions required by the Vision Transformer (ViT) model. This ensures that the patching mechanism of the ViT (16x16 patches in this case) works seamlessly, dividing the images into the correct number of patches for transformer-based processing.

2. **Enhanced Data Augmentations**: 
   - **RandAugment**:
     - Method: This augmentation policy applies a random combination of transformations such as rotation, brightness adjustment, and flipping, chosen from a predefined pool of operations.
     - Implementation: Integrated using the `RandAugment` class from `torchvision.transforms`.
     - Intuition: Augmentations simulate diverse scenarios in the dataset, enhancing model robustness to unseen variations in real-world applications.
   - **Random Erasing**:
     - Method: Randomly erases parts of an image during training by replacing selected regions with random pixel values.
     - Probability: Set to 0.25, meaning 25% of training images had a random region erased.
     - Intuition: Prevents the model from over-relying on specific regions of an image, encouraging it to learn more generalized features.

### Training Enhancements

1. **Mixup and CutMix**:
   - **Mixup**:
     - Method: Mixup blends two training examples and their labels, creating a synthetic training sample:  
       \[ \tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j \]  
       where \( \lambda \) is sampled from a Beta distribution.
     - Implementation: Integrated using the `Mixup` utility from the `timm` library.
     - Intuition: Mixup smoothens decision boundaries and reduces overfitting, as the model cannot rely on "hard" training labels.
   - **CutMix**:
     - Method: Similar to Mixup, but instead of blending the entire images, rectangular patches of one image replace patches in another. Labels are proportionally adjusted.
     - Implementation: Configured with probabilities for blending and patch placement using `timm.data.Mixup`.
     - Intuition: Encourages spatially aware feature learning, improving robustness to occlusions or corruptions.
   
2. **Stochastic Depth**:
   - Method: During training, randomly drops a subset of transformer blocks in each forward pass, controlled by a drop probability.
   - Implementation: Applied a drop probability of 0.1 to regularize deeper layers using `timm.layers.DropPath`.
   - Intuition: Mimics an ensemble effect by allowing the model to explore multiple sub-networks, reducing overfitting and improving generalization.

3. **AMP (Automatic Mixed Precision)**:
   - Method: Combines half-precision and full-precision computations dynamically during training.
   - Implementation: Enabled with `torch.amp.GradScaler` and `torch.cuda.amp.autocast`.
   - Intuition: Reduces GPU memory usage and accelerates training while maintaining model performance, especially useful for computationally intensive ViT models.

### Training Loop
- **Epochs**: Trained for up to 50 epochs but utilized early stopping after achieving peak validation accuracy (90.5%) at 10 epochs.
- **Batch Size**: Set to 128, optimized for GPU memory utilization.
- **Logging**: Metrics, including training and validation loss and accuracy, were logged using TensorBoard. Logging frequency was every 100 batches to balance granularity and performance overhead.

### Validation
- Standard Cross-Entropy loss was used during validation for hard-label accuracy computation. Unlike the training phase, which used soft-label losses (Mixup and CutMix), validation focused purely on the model's ability to classify with confidence in real-world scenarios.

---

### Layer Fine-Tuning Strategy

The experiment tested two configurations for fine-tuning:

1. **Fine-Tuning All Layers**:
   - In this setting, all layers of the ViT model were unfrozen, allowing gradient updates to modify the pretrained weights.
   - **Result**: Achieved a validation accuracy of 90.5%, demonstrating the ability of the model to adapt its internal representations to the Tiny ImageNet dataset.

2. **Fine-Tuning the Last Fully Connected Layer Only**:
   - In this setting, only the final classification head (Fully Connected Layer) was updated, while all transformer layers were frozen.
   - **Result**: Achieved a validation accuracy of 72.3%, indicating limited capacity to adapt the learned features to the new dataset.

**Analysis**:
- **Why Fine-Tuning All Layers Performed Better**:
  - The pretrained ViT model was trained on ImageNet, which shares some similarities with Tiny ImageNet but differs in scale and distribution.
  - Fine-tuning all layers allowed the model to adjust its intermediate representations to the specific features and patterns of the Tiny ImageNet dataset, leading to significantly better performance.
- **When to Fine-Tune Specific Layers**:
  - Fine-tuning specific layers, such as only the classification head, may suffice for tasks with highly similar datasets (e.g., same domain). However, for diverse datasets, fine-tuning more or all layers is generally necessary.

---

**Key Takeaway**: Fine-tuning the entire network maximized the model's adaptability to Tiny ImageNet, yielding superior performance. However, this comes at a higher computational cost compared to only tuning the last layer.

## Results

- **Validation Accuracy**: 90.5% after 10 epochs
- **Training Time**: Approximately 30 minutes per epoch on a single GPU
- **Best Model Saved**: Model checkpoint saved at `./models/best_vit_tiny_imagenet.pth`

## Key Insights
1. **Enhanced Augmentations**: The combination of RandAugment, Mixup, and CutMix improved generalization.
2. **Cosine Annealing**: Helped achieve smooth convergence with the learning rate.
3. **Pretrained Weights**: Accelerated convergence and boosted performance significantly.



---

**Repository Setup**:
The code for this implementation, including the preprocessing and training pipeline, is structured for easy reproducibility. Ensure you have the following dependencies installed:
- PyTorch
- torchvision
- timm
- tqdm

## Code
````markdown
```python

import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from timm import create_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # For progress bar
os.environ['HF_HOME'] = '/tmp/ygu2/hf_cache_custom'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/ygu2/hf_cache_custom/hub'

# Import Mixup and CutMix utilities from timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

# Optional: Import RandAugment for enhanced data augmentation
from torchvision.transforms import RandAugment

# Set CuDNN Benchmark for optimized performance
torch.backends.cudnn.benchmark = True

# Paths and Constants
data_dir = "./datasets/tiny-imagenet-200"
num_classes = 200
batch_size = 128  # Adjust based on GPU memory
num_epochs = 50  # Increased number of epochs for better convergence
learning_rate = 1e-4  # Lowered learning rate for fine-tuning
weight_decay = 0.01  # Adjusted weight decay
image_size = 384
log_interval = 100  # Log metrics every 100 batches

# Reorganize validation data
val_dir = os.path.join(data_dir, 'val')
val_images_dir = os.path.join(val_dir, 'images')
val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')

# Create a mapping from image filenames to their labels
val_img_dict = {}
with open(val_annotations_file, 'r') as f:
    for line in f.readlines():
        words = line.strip().split('\t')
        val_img_dict[words[0]] = words[1]

# Create directories for each class if they don't exist
for label in set(val_img_dict.values()):
    label_dir = os.path.join(val_images_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

# Move images into the corresponding label directories
for img_filename, label in val_img_dict.items():
    src = os.path.join(val_images_dir, img_filename)
    dst = os.path.join(val_images_dir, label, img_filename)
    if os.path.exists(src):
        shutil.move(src, dst)

# Data Augmentation and Transformations
transform_train = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    RandAugment(),  # Enhanced augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomErasing(p=0.25),
])

transform_test = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Load Datasets
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,  # Reduced from 8 to 2
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True
)

val_dataset = datasets.ImageFolder(val_images_dir, transform=transform_test)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,  # Reduced from 8 to 2
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True
)

# Create Vision Transformer (ViT) Model
model = create_model('vit_base_patch16_384', pretrained=True, num_classes=num_classes)

# Apply Stochastic Depth
from timm.layers import DropPath  # Updated import path

def apply_stochastic_depth(model, drop_prob):
    for module in model.modules():
        if isinstance(module, DropPath):
            module.drop_prob = drop_prob

apply_stochastic_depth(model, drop_prob=0.1)

# Unfreeze the entire model for fine-tuning
for param in model.parameters():
    param.requires_grad = True

# Use DataParallel for multiple GPUs if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)  # This will use all available GPUs

model = model.to(device)

# Mixup and CutMix
mixup_fn = Mixup(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    cutmix_minmax=None,
    prob=0.5,  # Reduced probability to allow some original images
    switch_prob=0.5,
    mode='batch',
    label_smoothing=0.1,
    num_classes=num_classes
)

# Loss, Optimizer, and Scheduler
criterion = SoftTargetCrossEntropy()  # For Mixup and CutMix

# Using SGD with momentum for better fine-tuning
optimizer = optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate,
    momentum=0.9,
    weight_decay=weight_decay
)

# Scheduler adjusted to steps per epoch
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Initialize AMP scaler for mixed precision
scaler = torch.amp.GradScaler(device='cuda')  # Updated instantiation

# Training and Validation Loop
writer = SummaryWriter()  # For TensorBoard logging

def train_one_epoch(epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    # Progress bar for training loop
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)
    for batch_idx, (images, labels) in enumerate(train_loader_tqdm):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Apply Mixup/CutMix
        images, labels = mixup_fn(images, labels)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        # Since labels are soft, calculate accuracy based on predicted class vs hard labels
        _, predicted = outputs.max(1)
        _, targets = labels.max(1)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar (accuracy in percentage)
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
            current_loss = loss.item()
            current_acc = 100. * correct / total
            train_loader_tqdm.set_postfix(loss=f"{current_loss:.4f}", accuracy=f"{current_acc:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total  # Multiply by 100 to get percentage
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")  # Acc in %

def validate(epoch):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    # Progress bar for validation loop
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False)

    criterion_val = nn.CrossEntropyLoss()  # Standard loss for validation

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader_tqdm):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion_val(outputs, labels)

            val_loss += loss.item() * images.size(0)
            total += labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar (accuracy in percentage)
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(val_loader):
                current_loss = loss.item()
                current_acc = 100. * correct / total
                val_loader_tqdm.set_postfix(loss=f"{current_loss:.4f}", accuracy=f"{current_acc:.2f}%")

    epoch_loss = val_loss / total
    epoch_acc = 100. * correct / total  # Multiply by 100 to get percentage
    writer.add_scalar('Loss/val', epoch_loss, epoch)
    writer.add_scalar('Accuracy/val', epoch_acc, epoch)
    print(f"Validation Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")  # Acc in %

    return epoch_acc

# Main Training Loop
best_acc = 0
for epoch in range(num_epochs):
    train_one_epoch(epoch)
    val_acc = validate(epoch)

    # Scheduler step
    scheduler.step()

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs('./models', exist_ok=True)
        # If using DataParallel, save the underlying model
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), './models/best_vit_tiny_imagenet.pth')
        else:
            torch.save(model.state_dict(), './models/best_vit_tiny_imagenet.pth')
        print(f"New best model saved with accuracy: {best_acc:.2f}%")

print("Training complete. Best validation accuracy:", best_acc)

writer.close()

```
````
