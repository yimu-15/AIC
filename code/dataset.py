"""
数据集定义 - 优化版
"""
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class NoisyDataset(Dataset):
    """噪声标签数据集"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []
        self.class_to_idx = {}

        # 扫描数据集
        for idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if not class_dir.is_dir():
                continue

            self.class_to_idx[class_dir.name] = idx

            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((str(img_path), idx))

        print(f"✓ Loaded {len(self.samples)} samples from {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回黑色图像
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label, index


class RandAugment:
    """RandAugment 数据增强"""
    def __init__(self, n, m):
        self.n = n  # 增强操作数量
        self.m = m  # 增强强度

        self.augment_list = [
            transforms.RandomAutocontrast(),
            transforms.RandomEqualize(),
            transforms.RandomPosterize(bits=4),
            transforms.RandomSolarize(threshold=128),
        ]

    def __call__(self, img):
        ops = np.random.choice(self.augment_list, self.n)
        for op in ops:
            img = op(img)
        return img


def get_train_transform(img_size=224, phase='warmup'):
    """训练数据增强"""
    if phase == 'warmup':
        # 预热阶段：温和增强
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    elif phase == 'filtering':
        # 过滤阶段：仅基础增强
        return transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    else:
        # 正常训练：强增强
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            RandAugment(n=2, m=10),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))
        ])


def get_val_transform(img_size=224):
    """验证数据增强"""
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])


def get_test_transform(img_size=224):
    """测试数据增强（TTA 基础）"""
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])


class MixupCutmix:
    """Mixup & CutMix 数据增强"""
    def __init__(self, mixup_alpha=0.4, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def __call__(self, images, labels):
        if np.random.rand() > self.prob:
            return images, None

        # 随机选择 Mixup 或 CutMix
        if np.random.rand() < 0.5:
            return self.mixup(images, labels)
        else:
            return self.cutmix(images, labels)

    def mixup(self, images, labels):
        """Mixup"""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a = labels
        labels_b = labels[index]

        return mixed_images, (labels_a, labels_b, lam)

    def cutmix(self, images, labels):
        """CutMix"""
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)

        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        _, _, h, w = images.size()
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        labels_a = labels
        labels_b = labels[index]

        return images, (labels_a, labels_b, lam)


class TestDataset(Dataset):
    """测试集数据集"""
    def __init__(self, test_dir, transform=None):
        self.test_dir = Path(test_dir)
        self.transform = transform

        self.image_paths = sorted(list(self.test_dir.glob('*.jpg')))
        print(f"✓ Loaded {len(self.image_paths)} test images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, img_path.name