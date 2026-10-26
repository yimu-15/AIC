"""
训练脚本 — 鲁棒修复版
=====================
关键修复:
  1. 移除双重前向传播（每个 batch 只调一次 model）
  2. 动态损失权重调度（前期强调 CE，后期引入蒸馏）
  3. 梯度范数监控（每 N 步打印各参数组梯度，便于调试）
  4. 集成 GMM 动态噪声分离 + CLIP 原型零样本评分 + 语义一致性 EMA
  5. LoRA Attention + VPT + 特征正则化全链路打通
  6. 显式 encode_image(normalize=False) 避免双归一化
"""
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
from torch.amp import autocast, GradScaler

from config import Config
from dataset import (
    NoisyDataset,
    get_train_transform,
    get_val_transform,
    MixupCutmix
)
from model import create_model, compute_clip_prototype_scores, load_frozen_reference_clip
from loss import CombinedLoss
from utils import (
    setup_seed,
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    EMAModel,
    compute_class_weights,
    get_cosine_schedule_with_warmup,
    log_gradient_norms,
)

# ═══════════════════════════════════════════════════════════════════
# 动态损失权重调度
# ═══════════════════════════════════════════════════════════════════

def get_dynamic_loss_weights(epoch, config):
    """
    前期强调 CE（让分类头 + LoRA 快速适应），
    后期引入蒸馏和特征正则化。

    三阶段:
      Stage 1 (epoch 1-5):   CE=0.7  distill=0.2  feature=0.1  reg=0.05
      Stage 2 (epoch 6-12):  CE=0.5  distill=0.35 feature=0.15 reg=0.05
      Stage 3 (epoch 13+):   CE=0.3  distill=0.55 feature=0.15 reg=0.05
    """
    start_epoch = config.DYNAMIC_WEIGHT_START_EPOCH
    mid_epoch = config.DYNAMIC_WEIGHT_MID_EPOCH

    if epoch < start_epoch:
        # 热身期: 强调 CE
        ce_w = 0.7
        distill_w = 0.2
        feature_w = 0.1
    elif epoch < mid_epoch:
        # 过渡期: 均衡
        frac = (epoch - start_epoch) / max(mid_epoch - start_epoch, 1)
        ce_w = 0.7 - 0.2 * frac
        distill_w = 0.2 + 0.15 * frac
        feature_w = 0.1 + 0.05 * frac
    else:
        # 稳定期: 蒸馏为主
        ce_w = 0.3
        distill_w = 0.55
        feature_w = 0.15

    reg_w = config.LOSS_FEATURE_REG_WEIGHT

    return ce_w, distill_w, feature_w, reg_w


# ═══════════════════════════════════════════════════════════════════
# 单 epoch 训练
# ═══════════════════════════════════════════════════════════════════

def train_one_epoch(epoch, model, ema_teacher, dataloader, criterion, optimizer,
                    scaler, scheduler, trust_filter, class_weights, config, logger,
                    mixup_cutmix=None, model_peer=None, reference_clip=None):
    """训练一个 epoch（修复版：单次前向 + 动态权重 + 梯度监控）"""
    model.train()
    if model_peer is not None:
        model_peer.train()

    losses_m = AverageMeter()
    accs_m = AverageMeter()

    # ── 动态损失权重 ──
    ce_w, distill_w, feature_w, reg_w = get_dynamic_loss_weights(epoch, config)

    # ── Trust Filter 权重因子 ──
    if epoch < config.TRUST_FILTER_START_EPOCH:
        trust_weight_factor = 0.0
    elif epoch < config.TRUST_FILTER_START_EPOCH + 7:
        trust_weight_factor = (epoch - config.TRUST_FILTER_START_EPOCH) / 7.0
    else:
        trust_weight_factor = 1.0

    use_coteaching = (config.USE_COTEACHING
                      and epoch <= config.COTEACHING_EPOCHS
                      and model_peer is not None)

    if use_coteaching:
        if config.COTEACHING_GRADUAL:
            forget_rate = config.COTEACHING_FORGET_RATE * (1 - epoch / config.COTEACHING_EPOCHS)
        else:
            forget_rate = config.COTEACHING_FORGET_RATE
        remember_rate = 1 - forget_rate

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} | "
                                 f"CE={ce_w:.2f} D={distill_w:.2f} F={feature_w:.2f} "
                                 f"Trust={trust_weight_factor:.2f}")

    for batch_idx, (images, labels, indices) in enumerate(pbar):
        images = images.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)

        # ── 伪标签修正 ──
        if epoch >= config.TRUST_FILTER_START_EPOCH and trust_filter is not None:
            corrected_labels = trust_filter.get_corrected_labels(indices, labels)
        else:
            corrected_labels = labels

        # ── 样本权重 ──
        if trust_weight_factor > 0:
            trust_weights = trust_filter.get_sample_weights(
                indices, class_weights, corrected_labels
            )
            sample_weights = ((1 - trust_weight_factor) * class_weights[corrected_labels]
                              + trust_weight_factor * trust_weights)
        else:
            sample_weights = class_weights[corrected_labels]

        # ── Mixup/CutMix（仅对高信任度样本） ──
        lam = None
        targets_a = None
        targets_b = None

        if epoch >= config.TRUST_FILTER_START_EPOCH and mixup_cutmix is not None:
            high_trust_mask = trust_filter.trust_scores[indices.cpu()] > 0.7
            if high_trust_mask.sum() > 1:
                clean_images = images[high_trust_mask]
                clean_labels_mix = corrected_labels[high_trust_mask]

                if len(clean_images) > 1:
                    clean_images, mixup_data = mixup_cutmix(clean_images, clean_labels_mix)
                    images[high_trust_mask] = clean_images

                    if isinstance(mixup_data, tuple):
                        targets_a = corrected_labels.clone()
                        targets_b = corrected_labels.clone()
                        ta_sub, tb_sub, lam = mixup_data
                        targets_a[high_trust_mask] = ta_sub
                        targets_b[high_trust_mask] = tb_sub

        # ── Co-teaching 样本选择 ──
        selected_mask = None
        if use_coteaching:
            with torch.no_grad():
                logits_1 = model(images)
                logits_2 = model_peer(images)

                loss_1 = F.cross_entropy(logits_1, corrected_labels, reduction='none')
                loss_2 = F.cross_entropy(logits_2, corrected_labels, reduction='none')

                num_remember = int(remember_rate * len(corrected_labels))
                _, ind_1 = torch.sort(loss_1)
                _, ind_2 = torch.sort(loss_2)

                selected_mask = torch.zeros(len(corrected_labels), dtype=torch.bool,
                                            device=images.device)
                selected_mask[ind_1[:num_remember]] = True
                selected_mask[ind_2[:num_remember]] = True

            if selected_mask.sum() > 0:
                images = images[selected_mask]
                corrected_labels = corrected_labels[selected_mask]
                sample_weights = sample_weights[selected_mask]
                if targets_a is not None:
                    targets_a = targets_a[selected_mask]
                    targets_b = targets_b[selected_mask]

        # ── 特征正则化锚定（独立冻结参考 CLIP） ──
        reference_features = None
        if config.USE_FEATURE_REG and reference_clip is not None:
            with torch.no_grad():
                with autocast(device_type='cuda', enabled=config.USE_AMP):
                    ref_feats = reference_clip.encode_image(images, normalize=True)
                reference_features = ref_feats.float()
                reference_features = F.normalize(reference_features, dim=-1)

        # ══════════════════════════════════════════════════════════
        # ★ 单次前向传播（修复：不再重复调用 model）
        # ══════════════════════════════════════════════════════════
        with autocast(device_type='cuda', enabled=config.USE_AMP):
            student_logits, student_features = model(images, return_features=True)

            with torch.no_grad():
                if ema_teacher is not None:
                    teacher_logits, teacher_features = ema_teacher(images, return_features=True)
                else:
                    teacher_logits = student_logits.detach()
                    teacher_features = student_features.detach()

            # ★ 动态损失权重传入 criterion
            loss, loss_dict = criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                targets=corrected_labels,
                lam=lam,
                targets_a=targets_a,
                targets_b=targets_b,
                sample_weights=sample_weights,
                student_features=student_features,
                teacher_features=teacher_features,
                reference_features=reference_features,
                # 动态权重覆盖
                ce_weight=ce_w,
                distill_weight=distill_w,
                feature_weight=feature_w,
                feature_reg_weight=reg_w,
            )

            loss = loss / config.ACCUMULATION_STEPS

        # ── 反向传播 ──
        scaler.scale(loss).backward()

        # ── 更新损失历史 ──
        with torch.no_grad():
            sample_losses = F.cross_entropy(
                student_logits.float(), corrected_labels, reduction='none'
            )
            if selected_mask is not None and selected_mask.sum() < len(indices):
                orig_indices = indices[selected_mask.cpu()]
            else:
                orig_indices = indices
            trust_filter.update_loss_history(orig_indices, sample_losses)

        # ── 梯度累积 ──
        if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            if ema_teacher is not None and config.USE_EMA:
                ema_teacher.update(model)

        # ── 梯度监控（前几个 batch 打印梯度范数，便于调试） ──
        if batch_idx == 0 and epoch <= 3:
            log_gradient_norms(model, logger, prefix=f"Epoch{epoch} Batch0")

        # ── 统计 ──
        with torch.no_grad():
            preds = student_logits.argmax(dim=1)
            acc = (preds == corrected_labels).float().mean()

        losses_m.update(loss_dict['total'], images.size(0))
        accs_m.update(acc.item(), images.size(0))

        pbar.set_postfix({
            'loss': f"{losses_m.avg:.4f}",
            'acc': f"{accs_m.avg:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })

        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            logger.info(
                f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
                f"Batch [{batch_idx+1}/{len(dataloader)}] "
                f"Loss: {losses_m.avg:.4f} Acc: {accs_m.avg:.4f} "
                f"| CE={ce_w:.2f} D={distill_w:.2f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    return losses_m.avg, accs_m.avg


# ═══════════════════════════════════════════════════════════════════
# 验证
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, dataloader, config):
    """验证"""
    model.eval()
    all_preds, all_labels = [], []

    for images, labels, _ in tqdm(dataloader, desc="Validating"):
        images = images.to(config.DEVICE, non_blocking=True)

        with autocast(device_type='cuda', enabled=config.USE_AMP):
            logits = model(images)

        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = (all_preds == all_labels).float().mean().item()

    return acc


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    config = Config()
    setup_seed(config.SEED)
    logger = setup_logging(config.LOG_DIR)

    logger.info("=" * 60)
    logger.info("TrustCLIP Training — Robust Edition")
    logger.info("=" * 60)
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Model: {config.MODEL_NAME} | Classes: {config.NUM_CLASSES}")
    logger.info(f"Batch: {config.BATCH_SIZE} × {config.ACCUMULATION_STEPS} = "
                f"{config.BATCH_SIZE * config.ACCUMULATION_STEPS}")
    logger.info(f"LoRA: rank={config.LORA_RANK} alpha={config.LORA_ALPHA} "
                f"attn={'ON' if getattr(config, 'USE_ATTENTION_LORA', False) else 'OFF'}")
    logger.info(f"VPT: {'ON' if getattr(config, 'USE_VPT', False) else 'OFF'}")
    logger.info(f"GMM Scoring: {'ON' if config.USE_GMM_SCORING else 'OFF'}")
    logger.info(f"CLIP Proto Scoring: {'ON' if config.USE_CLIP_PROTOTYPE_SCORING else 'OFF'}")
    logger.info(f"Feature Reg: {'ON' if config.USE_FEATURE_REG else 'OFF'}")
    logger.info(f"Dynamic Weights: ON (start={config.DYNAMIC_WEIGHT_START_EPOCH}, "
                f"mid={config.DYNAMIC_WEIGHT_MID_EPOCH})")

    # ═════════════════════════════════════════════════════════════
    # [1/5] 数据加载
    # ═════════════════════════════════════════════════════════════
    logger.info("\n[1/5] Loading datasets...")

    train_dataset = NoisyDataset(
        config.TRAIN_DIR,
        transform=get_train_transform(config.IMG_SIZE, phase='warmup')
    )

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False,
        drop_last=True,  # 避免最后一个 batch size 不一致导致 BN 问题
    )

    val_dataset_transform = NoisyDataset(
        config.TRAIN_DIR,
        transform=get_val_transform(config.IMG_SIZE)
    )
    val_subset.dataset = val_dataset_transform

    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    logger.info(f"  Train: {len(train_subset)} | Val: {len(val_subset)}")

    # ═════════════════════════════════════════════════════════════
    # [2/5] 创建模型
    # ═════════════════════════════════════════════════════════════
    logger.info("\n[2/5] Creating models...")
    model, ema_teacher, trust_filter = create_model(config, len(train_dataset))
    model = model.to(config.DEVICE)

    if ema_teacher is not None:
        ema_teacher = ema_teacher.to(config.DEVICE)

    # Co-teaching peer
    model_peer = None
    if config.USE_COTEACHING:
        from model import CLIPWithLoRA
        model_peer = CLIPWithLoRA(config).to(config.DEVICE)
        logger.info("  ✓ Co-teaching peer created")

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params: {total_p:,}")
    logger.info(f"  Trainable: {trainable_p:,} ({100*trainable_p/total_p:.2f}%)")

    # CLIP 原型零样本评分（独立冻结 CLIP，训练全程不变）
    if config.USE_CLIP_PROTOTYPE_SCORING:
        logger.info("\n[Prototype Scoring] Computing CLIP prototype scores...")
        clip_proto_scores = compute_clip_prototype_scores(config, train_dataset)
        trust_filter.set_clip_proto_scores(clip_proto_scores)

    # 特征正则化锚点
    reference_clip = None
    if config.USE_FEATURE_REG:
        reference_clip = load_frozen_reference_clip(config).to(config.DEVICE)

    # ═════════════════════════════════════════════════════════════
    # [3/5] 类别权重
    # ═════════════════════════════════════════════════════════════
    logger.info("\n[3/5] Computing class weights...")
    class_weights = compute_class_weights(
        train_dataset, config.NUM_CLASSES, beta=config.CLASS_BALANCE_BETA
    ).to(config.DEVICE)

    # ═════════════════════════════════════════════════════════════
    # [4/5] 优化器
    # ═════════════════════════════════════════════════════════════
    logger.info("\n[4/5] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=config.BETAS
    )

    optimizer_peer = None
    if model_peer is not None:
        optimizer_peer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model_peer.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.BETAS
        )

    total_steps = len(train_loader) // config.ACCUMULATION_STEPS * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, min_lr=config.MIN_LR
    )

    scheduler_peer = None
    if optimizer_peer is not None:
        scheduler_peer = get_cosine_schedule_with_warmup(
            optimizer_peer, warmup_steps, total_steps, min_lr=config.MIN_LR
        )

    criterion = CombinedLoss(config).to(config.DEVICE)
    scaler = GradScaler('cuda') if config.USE_AMP else None
    scaler_peer = GradScaler('cuda') if (config.USE_AMP and model_peer is not None) else None

    mixup_cutmix = MixupCutmix(
        mixup_alpha=config.MIXUP_ALPHA,
        cutmix_alpha=config.CUTMIX_ALPHA,
        prob=config.MIXUP_PROB
    )

    # ═════════════════════════════════════════════════════════════
    # [5/5] 训练
    # ═════════════════════════════════════════════════════════════
    logger.info("\n[5/5] Training...")
    logger.info(f"  Dynamic weights: epochs 1-{config.DYNAMIC_WEIGHT_START_EPOCH-1} "
                f"(CE-heavy) → {config.DYNAMIC_WEIGHT_START_EPOCH}-"
                f"{config.DYNAMIC_WEIGHT_MID_EPOCH-1} (balanced) → "
                f"{config.DYNAMIC_WEIGHT_MID_EPOCH}+ (distill-heavy)")

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{config.NUM_EPOCHS}")
        logger.info(f"{'='*60}")

        # ── Trust Filter 评分更新 ──
        if epoch == config.TRUST_FILTER_START_EPOCH:
            logger.info("\n[Trust] Computing initial trust scores...")
            train_dataset.transform = get_train_transform(config.IMG_SIZE, phase='filtering')
            trust_filter.compute_trust_scores(model, train_loader, config.DEVICE, config)

        if epoch > config.TRUST_FILTER_START_EPOCH and epoch % config.TRUST_UPDATE_FREQUENCY == 0:
            logger.info("\n[Trust] Updating trust scores...")
            trust_filter.compute_trust_scores(model, train_loader, config.DEVICE, config)

        # ── GMM 动态噪声评分 ──
        if config.USE_GMM_SCORING:
            if epoch == config.GMM_START_EPOCH or (
                epoch > config.GMM_START_EPOCH
                and epoch % config.GMM_UPDATE_FREQUENCY == 0
            ):
                logger.info("\n[GMM] Fitting Gaussian Mixture Model on loss history...")
                trust_filter.fit_gmm()

        # ── 伪标签 ──
        if config.USE_PSEUDO_LABEL and epoch >= config.PSEUDO_LABEL_START_EPOCH:
            if (epoch == config.PSEUDO_LABEL_START_EPOCH
                    or epoch % config.PSEUDO_LABEL_UPDATE_FREQ == 0):
                logger.info("\n[Pseudo Label] Updating pseudo labels...")
                trust_filter.update_pseudo_labels(
                    model, train_loader, config.DEVICE,
                    threshold=config.PSEUDO_LABEL_THRESHOLD
                )

        # ── 训练 ──
        # 恢复强增强 transform
        train_dataset.transform = get_train_transform(config.IMG_SIZE, phase='train')

        train_loss, train_acc = train_one_epoch(
            epoch=epoch,
            model=model,
            ema_teacher=ema_teacher,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            trust_filter=trust_filter,
            class_weights=class_weights,
            config=config,
            logger=logger,
            mixup_cutmix=mixup_cutmix if epoch >= config.TRUST_FILTER_START_EPOCH else None,
            model_peer=model_peer,
            reference_clip=reference_clip,
        )

        # Co-teaching peer
        if model_peer is not None and optimizer_peer is not None:
            train_one_epoch(
                epoch=epoch,
                model=model_peer,
                ema_teacher=None,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer_peer,
                scaler=scaler_peer,
                scheduler=scheduler_peer,
                trust_filter=trust_filter,
                class_weights=class_weights,
                config=config,
                logger=logger,
                mixup_cutmix=mixup_cutmix if epoch >= config.TRUST_FILTER_START_EPOCH else None,
                model_peer=model,
                reference_clip=reference_clip,
            )

        # ── 验证 ──
        val_acc = validate(model, val_loader, config)
        ema_val_acc = 0.0
        if ema_teacher is not None:
            ema_val_acc = validate(ema_teacher.teacher, val_loader, config)
            logger.info(f"Val Acc: {val_acc:.4f} | EMA Val Acc: {ema_val_acc:.4f}")
        else:
            logger.info(f"Val Acc: {val_acc:.4f}")

        current_acc = max(val_acc, ema_val_acc)
        ce_w, distill_w, feature_w, reg_w = get_dynamic_loss_weights(epoch, config)
        logger.info(f"  Loss weights: CE={ce_w:.2f} Distill={distill_w:.2f} "
                    f"Feature={feature_w:.2f} Reg={reg_w:.4f}")

        # ── 早停 + 保存 ──
        if current_acc > best_acc:
            best_acc = current_acc
            patience_counter = 0

            save_model = (ema_teacher.teacher
                          if ema_val_acc > val_acc and ema_teacher is not None
                          else model)
            best_ckpt = Path(config.CHECKPOINT_DIR) / "best_model.pt"
            save_checkpoint(save_model, optimizer, scaler, epoch, best_acc, None, best_ckpt)
            logger.info(f"✓ Best model saved! (Acc: {best_acc:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"\n⚠ Early stopping at epoch {epoch}! Best Acc: {best_acc:.4f}")
                break

        # 定期保存
        if epoch % config.SAVE_INTERVAL == 0:
            ckpt_path = Path(config.CHECKPOINT_DIR) / f"epoch_{epoch}.pt"
            save_checkpoint(model, optimizer, scaler, epoch, current_acc, ema_teacher, ckpt_path)

        # 滚动损失历史缓冲区
        trust_filter.advance_epoch()

    logger.info(f"\n{'='*60}")
    logger.info(f"Training completed! Best Acc: {best_acc:.4f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
