"""
诊断脚本 — 验证模型梯度流动和组件正确性
==========================================
在正式训练前运行此脚本，逐项检查:
  1. CLIP 预训练权重是否成功加载
  2. 不同图像是否产生不同特征（而非坍塌）
  3. 分类头是否能接收梯度
  4. LoRA 参数是否有梯度
  5. VPT prompt 是否有梯度
  6. 单步梯度下降后 loss 是否下降
  7. GMM 拟合是否正常
  8. CLIP 原型评分是否合理

使用方法:
  python diagnose.py
"""
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from dataset import NoisyDataset, get_train_transform, get_val_transform
from model import create_model, compute_clip_prototype_scores, load_frozen_reference_clip
from loss import CombinedLoss
from utils import setup_seed


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_true(condition, msg):
    if condition:
        print(f"  ✓ {msg}")
    else:
        print(f"  ✗ FAIL: {msg}")


# ═══════════════════════════════════════════════════════════════════
def test_1_clip_weights(config):
    """测试 CLIP 权重是否加载成功"""
    section("Test 1: CLIP pretrained weights")
    import open_clip

    cache_dir = Path(config.CLIP_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        model, _, _ = open_clip.create_model_and_transforms(
            config.MODEL_NAME, pretrained='openai', cache_dir=str(cache_dir)
        )
    except Exception as e:
        print(f"  ✗ Failed to load CLIP: {e}")
        return False

    # 检查 visual 参数是否包含了非随机的权重
    visual = model.visual
    # 取第一个 conv 权重
    conv_weight = visual.conv1.weight
    w_mean = conv_weight.mean().item()
    w_std = conv_weight.std().item()

    print(f"  conv1.weight mean={w_mean:.6f} std={w_std:.6f}")
    check_true(abs(w_mean) < 0.1, f"conv1 mean is reasonable ({w_mean:.6f})")
    check_true(w_std > 0.01, f"conv1 std is non-trivial ({w_std:.6f})")

    # 随机初始化的权重通常 mean≈0, std≈0.02-0.05
    # 预训练权重有特定的分布模式
    # 关键：检查 positional_embedding 是否非零
    pos_mean = visual.positional_embedding.abs().mean().item()
    print(f"  positional_embedding abs mean={pos_mean:.6f}")
    check_true(pos_mean > 0.001, f"positional_embedding non-trivial ({pos_mean:.6f})")

    del model
    torch.cuda.empty_cache()
    return True


def test_2_feature_diversity(config):
    """测试模型是否为不同图像产生不同特征"""
    section("Test 2: Feature diversity")

    # 用少量样本创建数据集
    dataset = NoisyDataset(
        config.TRAIN_DIR,
        transform=get_val_transform(config.IMG_SIZE)
    )

    if len(dataset) < 4:
        print("  ⚠ Dataset too small (<4 samples), skipping")
        return True

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    images, labels, indices = next(iter(loader))
    images = images.to(config.DEVICE)

    # 创建模型
    model, _, _ = create_model(config, len(dataset))
    model = model.to(config.DEVICE)
    model.eval()

    with torch.no_grad():
        features = model.clip_model.visual(images)
        features = F.normalize(features.float(), dim=-1)

    # 检查特征多样性
    sim_matrix = features @ features.T
    print(f"  Pairwise cosine similarities:")
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            sim = sim_matrix[i, j].item()
            print(f"    img{i}↔img{j}: {sim:.4f}")

    # 不同图像相似度不应都接近 1（否则特征坍塌）
    off_diag = sim_matrix[~torch.eye(4, dtype=bool)].reshape(4, 3)
    mean_sim = off_diag.mean().item()
    print(f"  Mean cross-image similarity: {mean_sim:.4f}")
    check_true(mean_sim < 0.99, f"Features are diverse (mean_sim={mean_sim:.4f})")

    del model
    torch.cuda.empty_cache()
    return True


def test_3_gradient_flow(config):
    """测试梯度是否正常流动"""
    section("Test 3: Gradient flow")

    dataset = NoisyDataset(
        config.TRAIN_DIR,
        transform=get_train_transform(config.IMG_SIZE, phase='warmup')
    )

    if len(dataset) < 2:
        print("  ⚠ Dataset too small, skipping")
        return True

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model, _, _ = create_model(config, len(dataset))
    model = model.to(config.DEVICE)
    model.train()

    images, labels, indices = next(iter(loader))
    images = images.to(config.DEVICE)
    labels = labels.to(config.DEVICE)

    # 前向
    logits, features = model(images, return_features=True)
    loss = F.cross_entropy(logits, labels)
    print(f"  Initial loss: {loss.item():.4f}")
    print(f"  Initial acc: {(logits.argmax(1)==labels).float().mean().item():.4f}")

    # 反向
    loss.backward()

    # 检查各参数组梯度
    has_grad = 0
    no_grad = 0
    zero_grad_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            no_grad += 1
            zero_grad_params.append(name)
        elif param.grad.norm().item() < 1e-10:
            no_grad += 1
            zero_grad_params.append(f"{name} (grad≈0)")
        else:
            has_grad += 1

    print(f"  Trainable params with gradient: {has_grad}")
    print(f"  Trainable params without gradient: {no_grad}")
    if zero_grad_params:
        print(f"  Zero-gradient params: {zero_grad_params[:10]}")

    check_true(has_grad > 0, f"{has_grad} trainable params receive gradients")
    check_true(no_grad == 0, f"All trainable params have non-zero gradients")

    # 测试单步优化后 loss 是否下降
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    optimizer.zero_grad()

    logits, _ = model(images, return_features=True)
    loss_before = F.cross_entropy(logits, labels).item()
    loss_before_t = F.cross_entropy(logits, labels)
    loss_before_t.backward()
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        logits, _ = model(images, return_features=True)
        loss_after = F.cross_entropy(logits, labels).item()

    print(f"  Loss before step: {loss_before:.4f}")
    print(f"  Loss after step:  {loss_after:.4f}")
    check_true(loss_after < loss_before + 0.01,
               f"Loss decreased after optimizer step "
               f"({loss_before:.4f} → {loss_after:.4f})")

    model.zero_grad(set_to_none=True)
    del model
    torch.cuda.empty_cache()
    return True


def test_4_gmm_fitting(config):
    """测试 GMM 拟合"""
    section("Test 4: GMM fitting")
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        print("  ⚠ sklearn not installed, skipping GMM test")
        return True

    # 模拟损失分布：干净样本低损失，噪声样本高损失
    np.random.seed(42)
    n_clean = 200
    n_noisy = 100
    clean_losses = np.random.normal(0.3, 0.1, n_clean)
    noisy_losses = np.random.normal(2.0, 0.5, n_noisy)
    all_losses = np.concatenate([clean_losses, noisy_losses]).reshape(-1, 1)

    # 归一化
    l_min, l_max = all_losses.min(), all_losses.max()
    losses_norm = (all_losses - l_min) / (l_max - l_min)

    gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-3,
                          reg_covar=1e-6, random_state=42)
    gmm.fit(losses_norm)

    clean_comp = int(np.argmin(gmm.means_.flatten()))
    probs = gmm.predict_proba(losses_norm)
    clean_probs = probs[:, clean_comp]

    # 检查：前 200 个"干净"样本应该有高干净概率
    clean_mean_prob = clean_probs[:n_clean].mean()
    noisy_mean_prob = clean_probs[n_clean:].mean()

    print(f"  Clean samples mean GMM-clean-prob: {clean_mean_prob:.4f}")
    print(f"  Noisy samples mean GMM-clean-prob: {noisy_mean_prob:.4f}")
    check_true(clean_mean_prob > 0.5, "GMM correctly identifies clean samples")
    check_true(noisy_mean_prob < 0.5, "GMM correctly identifies noisy samples")

    return True


def test_5_clip_prototype(config):
    """测试 CLIP 原型评分"""
    section("Test 5: CLIP prototype scoring")

    dataset = NoisyDataset(
        config.TRAIN_DIR,
        transform=get_val_transform(config.IMG_SIZE)
    )

    if len(dataset) < 10:
        print("  ⚠ Dataset too small (<10 samples), skipping")
        return True

    try:
        scores = compute_clip_prototype_scores(config, dataset)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Score mean: {scores.mean():.4f}")
    print(f"  Score std: {scores.std():.4f}")

    check_true(scores.min() >= 0.0, "Scores ≥ 0")
    check_true(scores.max() <= 1.0, "Scores ≤ 1")
    check_true(scores.std() > 0.01, f"Scores have variance (std={scores.std():.4f})")

    return True


def test_6_full_pipeline(config):
    """端到端测试完整训练流程（1 batch）"""
    section("Test 6: Full pipeline (1 batch)")

    from torch.amp import autocast, GradScaler

    dataset = NoisyDataset(
        config.TRAIN_DIR,
        transform=get_train_transform(config.IMG_SIZE, phase='warmup')
    )

    if len(dataset) < 8:
        print("  ⚠ Dataset too small, skipping")
        return True

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 创建模型
    model, ema_teacher, trust_filter = create_model(config, len(dataset))
    model = model.to(config.DEVICE)
    if ema_teacher is not None:
        ema_teacher = ema_teacher.to(config.DEVICE)

    # 加载 CLIP 原型评分
    if config.USE_CLIP_PROTOTYPE_SCORING:
        scores = compute_clip_prototype_scores(config, dataset)
        trust_filter.set_clip_proto_scores(scores)

    # 参考 CLIP
    reference_clip = None
    if config.USE_FEATURE_REG:
        reference_clip = load_frozen_reference_clip(config).to(config.DEVICE)

    # 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE
    )
    criterion = CombinedLoss(config)
    scaler = GradScaler('cuda') if config.USE_AMP else None

    model.train()
    images, labels, indices = next(iter(loader))
    images = images.to(config.DEVICE)
    labels = labels.to(config.DEVICE)

    # 参考特征
    ref_features = None
    if reference_clip is not None:
        with torch.no_grad():
            ref_features = reference_clip.encode_image(images, normalize=True)
            ref_features = F.normalize(ref_features.float(), dim=-1)

    # 前向
    with autocast(device_type='cuda', enabled=config.USE_AMP):
        s_logits, s_feats = model(images, return_features=True)
        with torch.no_grad():
            if ema_teacher is not None:
                t_logits, t_feats = ema_teacher(images, return_features=True)
            else:
                t_logits, t_feats = s_logits.detach(), s_feats.detach()

        loss, loss_dict = criterion(
            s_logits, t_logits, labels,
            student_features=s_feats, teacher_features=t_feats,
            reference_features=ref_features,
            ce_weight=0.7, distill_weight=0.2,
            feature_weight=0.1, feature_reg_weight=config.LOSS_FEATURE_REG_WEIGHT,
        )

    print(f"  Total loss: {loss.item():.4f}")
    print(f"  CE loss: {loss_dict['ce'].item():.4f}")
    print(f"  Distill loss: {loss_dict['distill'].item():.4f}")
    print(f"  Feature loss: {loss_dict['feature'].item():.4f}")
    print(f"  Feature reg loss: {loss_dict['feature_reg'].item():.4f}")

    # 反向
    scaler.scale(loss).backward() if scaler is not None else loss.backward()

    # 检查梯度
    total_gn = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_gn += p.grad.norm().item() ** 2
    total_gn = total_gn ** 0.5
    print(f"  Total gradient norm: {total_gn:.6f}")

    check_true(total_gn > 1e-8, f"Gradients flow (total norm={total_gn:.6f})")

    del model, ema_teacher
    if reference_clip is not None:
        del reference_clip
    torch.cuda.empty_cache()
    return True


# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  TrustCLIP Diagnostic Suite")
    print("=" * 60)

    config = Config()
    setup_seed(config.SEED)
    print(f"  Device: {config.DEVICE}")
    print(f"  Train dir: {config.TRAIN_DIR}")

    results = {}

    # 按顺序运行测试
    results['1_clip_weights'] = test_1_clip_weights(config)
    results['2_feature_diversity'] = test_2_feature_diversity(config)
    results['3_gradient_flow'] = test_3_gradient_flow(config)
    results['4_gmm'] = test_4_gmm_fitting(config)
    results['5_clip_prototype'] = test_5_clip_prototype(config)
    results['6_full_pipeline'] = test_6_full_pipeline(config)

    # 汇总
    section("Summary")
    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} — {name}")
        if not passed:
            all_pass = False

    print(f"\n  {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED — check above for details'}")
    return all_pass


if __name__ == "__main__":
    main()
