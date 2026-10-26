"""
独立评估脚本 - 高信任样本准确率分析
用法: python eval_trust.py
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import numpy as np

from config import Config
from dataset import NoisyDataset, get_train_transform, get_val_transform
from model import create_model, TrustFilter
from utils import load_checkpoint, setup_seed, setup_logging

@torch.no_grad()
def compute_trust_scores_all(model, dataloader, num_samples, num_classes, device, config):
    """计算所有样本的信任分数（不依赖训练时的TrustFilter）"""
    model.eval()
    trust_filter = TrustFilter(num_samples, num_classes, config)
    # 临时替换 compute_trust_scores 使用全量 dataloader
    all_features, all_labels, all_indices = [], [], []

    for images, labels, indices in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            _, feats = model(images, return_features=True)
        all_features.append(feats.cpu())
        all_labels.append(labels)
        all_indices.append(indices)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    all_indices = torch.cat(all_indices)

    # 复用 TrustFilter 的计算逻辑（类内相似度）
    all_features = F.normalize(all_features, dim=-1)
    scores = torch.zeros(num_samples)
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() <= 1:
            continue
        class_feats = all_features[mask]
        sim = class_feats @ class_feats.T
        sim.fill_diagonal_(0)
        mean_sim = sim.sum(dim=1) / (mask.sum() - 1)
        scores[all_indices[mask]] = mean_sim

    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    trust_filter.trust_scores = scores
    return trust_filter

@torch.no_grad()
def evaluate_with_trust(model, dataloader, trust_filter, threshold, device, config):
    """在验证集上评估信任分数 > threshold 的样本准确率"""
    model.eval()
    correct, total = 0, 0
    for images, labels, indices in tqdm(dataloader, desc=f"Eval trust>{threshold}"):
        mask = trust_filter.trust_scores[indices] > threshold
        if mask.sum() == 0:
            continue
        images = images[mask].to(device)
        lbls = labels[mask].to(device)
        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == lbls).sum().item()
        total += len(lbls)
    return correct / total if total > 0 else 0.0

def main():
    config = Config()
    setup_seed(config.SEED)
    logger = setup_logging(config.LOG_DIR)
    device = config.DEVICE

    # 加载训练好的最佳模型
    print("Loading best model...")
    model, _, _ = create_model(config, num_samples=0)  # num_samples 任意，后面重建 TrustFilter
    checkpoint = torch.load(Path(config.CHECKPOINT_DIR) / "best_model.pt", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 准备全量训练集（包含验证集那部分，因为我们想评估验证集上的信任分数）
    # 注意：这里我们重新创建一个包含所有样本的数据集，并用 filtering transform
    full_dataset = NoisyDataset(config.TRAIN_DIR, transform=get_train_transform(config.IMG_SIZE, phase='filtering'))
    full_loader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE*2, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)

    # 计算全量样本的信任分数
    print("Computing trust scores on all training data...")
    trust_filter = compute_trust_scores_all(model, full_loader, len(full_dataset),
                                            config.NUM_CLASSES, device, config)

    # 划分验证集（和训练时一样的划分方式，或者我们直接读取保存的 indices？）
    # 简便起见：我们用随机种子重新划分一次，但要保证与训练时完全一致。
    # 由于训练时用 generator=torch.Generator().manual_seed(config.SEED)，我们复现划分。
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_sub, val_sub = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    # 为验证集应用 val_transform
    val_dataset = NoisyDataset(config.TRAIN_DIR, transform=get_val_transform(config.IMG_SIZE))
    val_sub.dataset = val_dataset
    val_loader = DataLoader(val_sub, batch_size=config.BATCH_SIZE*2, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    # 评估不同信任阈值下的准确率
    thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("\nTrust threshold -> Val accuracy (number of high-trust samples in val):")
    for th in thresholds:
        acc = evaluate_with_trust(model, val_loader, trust_filter, th, device, config)
        # 统计验证集中信任分数 > th 的样本数
        val_indices = torch.tensor(val_sub.indices)
        high_trust_count = (trust_filter.trust_scores[val_indices] > th).sum().item()
        print(f"  Threshold {th:.1f}: Acc = {acc:.4f} | #High-trust in val = {high_trust_count}/{len(val_sub)}")

    # 整体准确率（无筛选）
    acc_all = evaluate_with_trust(model, val_loader, trust_filter, threshold=-1.0, device=device, config=config)
    print(f"\nOverall val acc (no filter): {acc_all:.4f}")

if __name__ == "__main__":
    main()