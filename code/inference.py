"""
推理脚本 - 优化版（支持 TTA）
"""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

from config import Config
from dataset import TestDataset, get_test_transform
from model import CLIPWithLoRA


class TTATransforms:
    """Test-Time Augmentation"""
    def __init__(self, img_size=224):
        from torchvision import transforms
        
        self.transforms = [
            # 原始
            transforms.Compose([
                transforms.Resize(img_size + 32),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ]),
            
            # 水平翻转
            transforms.Compose([
                transforms.Resize(img_size + 32),
                transforms.CenterCrop(img_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ]),
            
            # 左上角裁剪
            transforms.Compose([
                transforms.Resize(img_size + 32),
                transforms.Lambda(lambda img: transforms.functional.crop(img, 0, 0, img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ]),
            
            # 右上角裁剪
            transforms.Compose([
                transforms.Resize(img_size + 32),
                transforms.Lambda(lambda img: transforms.functional.crop(img, 0, 32, img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ]),
            
            # 左下角裁剪
            transforms.Compose([
                transforms.Resize(img_size + 32),
                transforms.Lambda(lambda img: transforms.functional.crop(img, 32, 0, img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ]),
        ]
    
    def __len__(self):
        return len(self.transforms)
    
    def __getitem__(self, idx):
        return self.transforms[idx]


@torch.no_grad()
def predict_with_tta(model, image, tta_transforms, device):
    """使用 TTA 进行预测"""
    from PIL import Image
    
    all_logits = []
    
    for transform in tta_transforms:
        # 转换图像
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # 预测
        with torch.cuda.amp.autocast():
            logits = model(img_tensor)
        
        all_logits.append(logits)
    
    # 平均 logits
    avg_logits = torch.stack(all_logits).mean(dim=0)
    
    return avg_logits


@torch.no_grad()
def inference(args):
    """推理主函数"""
    config = Config()
    
    print("=" * 50)
    print("TrustCLIP Inference")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Dir: {args.test_dir}")
    print(f"Output: {args.output}")
    print(f"Use TTA: {args.use_tta}")
    print(f"Batch Size: {args.batch_size}")
    
    # 加载模型
    print("\n[1/3] Loading model...")
    model = CLIPWithLoRA(config).to(config.DEVICE)
    
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    
    # 尝试加载 EMA 模型（如果有）
    if 'ema_model_state_dict' in checkpoint:
        print("  ✓ Loading EMA model weights")
        model.load_state_dict(checkpoint['ema_model_state_dict'])
    else:
        print("  ✓ Loading student model weights")
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    print(f"  Model loaded from epoch {checkpoint['epoch']} (Acc: {checkpoint['acc']:.4f})")
    
    # 准备数据
    print("\n[2/3] Preparing data...")
    
    if args.use_tta:
        # TTA 模式：逐张处理
        from PIL import Image
        test_images = sorted(Path(args.test_dir).glob('*.jpg'))
        tta_transforms = TTATransforms(config.IMG_SIZE)
        
        print(f"  ✓ Loaded {len(test_images)} test images")
        print(f"  ✓ TTA augmentations: {len(tta_transforms)}")
        
        all_predictions = []
        all_filenames = []
        
        print("\n[3/3] Running inference with TTA...")
        for img_path in tqdm(test_images, desc="Predicting"):
            image = Image.open(img_path).convert('RGB')
            
            logits = predict_with_tta(model, image, tta_transforms, config.DEVICE)
            pred = logits.argmax(dim=1).item()
            
            all_predictions.append(pred)
            all_filenames.append(img_path.name)
    
    else:
        # 标准模式：批量处理
        test_dataset = TestDataset(args.test_dir, transform=get_test_transform(config.IMG_SIZE))
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        all_predictions = []
        all_filenames = []
        
        print("\n[3/3] Running inference...")
        for images, filenames in tqdm(test_loader, desc="Predicting"):
            images = images.to(config.DEVICE)
            
            with torch.cuda.amp.autocast():
                logits = model(images)
            
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_predictions.extend(preds)
            all_filenames.extend(filenames)
    
    # 保存结果
    print("\n[4/4] Saving results...")
    df = pd.DataFrame({
        'image': all_filenames,
        'label': all_predictions
    })
    
    # 按文件名排序
    df['image_id'] = df['image'].str.extract(r'(\d+)').astype(int)
    df = df.sort_values('image_id').drop('image_id', axis=1)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Predictions saved to {output_path}")
    print(f"  Total predictions: {len(df)}")
    print(f"  Label distribution:")
    print(df['label'].value_counts().head(10))


def main():
    parser = argparse.ArgumentParser(description="TrustCLIP Inference Script")
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )
    
    parser.add_argument(
        '--test_dir',
        type=str,
        required=True,
        help='Path to test images directory'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='submission.csv',
        help='Output CSV file path (default: submission.csv)'
    )
    
    parser.add_argument(
        '--use_tta',
        action='store_true',
        help='Enable Test-Time Augmentation (slower but more accurate)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for inference (default: 128)'
    )
    
    args = parser.parse_args()
    
    inference(args)


if __name__ == '__main__':
    main()
