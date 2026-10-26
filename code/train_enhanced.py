"""
增强训练脚本 - GMM + Visual Prompt + 特征正则化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from torch.amp import autocast, GradScaler

from config_enhanced import Config
from dataset import NoisyDataset, get_train_transform, get_val_transform, MixupCutmix
from model_enhanced import CLIPEnhanced
from loss import CombinedLoss
from utils import (setup_seed, setup_logging, save_checkpoint,
                   AverageMeter, EMAModel, compute_class_weights,
                   get_cosine_schedule_with_warmup)
from gmm_filter import GMMFilter

def feature_regularization_loss(s_feats, t_feats):
    s_n = F.normalize(s_feats, dim=-1)
    t_n = F.normalize(t_feats, dim=-1)
    return 1 - (s_n * t_n).sum(dim=-1).mean()

def train_one_epoch(epoch, model, ema_teacher, dataloader, criterion,
                    optimizer, scaler, scheduler, gmm_filter, class_weights,
                    config, logger, frozen_clip=None):
    model.train()
    losses_m = AverageMeter()
    accs_m = AverageMeter()

    gmm_w = gmm_filter.clean_probs if (config.USE_GMM_FILTER and epoch > config.GMM_WARMUP_EPOCHS) else torch.ones(len(gmm_filter.clean_probs))

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}")
    for imgs, labels, idxs in pbar:
        imgs = imgs.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)

        cw = class_weights[labels]
        gw = gmm_w[idxs].to(config.DEVICE)
        sample_w = cw * gw
        sample_w = sample_w / sample_w.mean()

        with autocast(device_type='cuda', enabled=config.USE_AMP):
            s_logits, s_feats = model(imgs, return_features=True)
            if ema_teacher:
                t_logits, t_feats = ema_teacher(imgs, return_features=True)
            else:
                t_logits, t_feats = s_logits.detach(), s_feats.detach()

            loss, loss_dict = criterion(s_logits, t_logits, labels,
                                        sample_weights=sample_w,
                                        student_features=s_feats,
                                        teacher_features=t_feats)

            # 特征正则化
            if config.LOSS_FEATURE_WEIGHT > 0 and frozen_clip is not None:
                with torch.no_grad():
                    froz_feats = frozen_clip.encode_image(imgs)
                    froz_feats = froz_feats / froz_feats.norm(dim=-1, keepdim=True)
                loss = loss + config.LOSS_FEATURE_WEIGHT * feature_regularization_loss(s_feats, froz_feats)

            loss = loss / config.ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        with torch.no_grad():
            per_loss = F.cross_entropy(s_logits, labels, reduction='none')
        gmm_filter.update(per_loss.detach().cpu(), idxs)

        if (pbar.n + 1) % config.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            if ema_teacher and config.USE_EMA:
                ema_teacher.update(model)

        with torch.no_grad():
            acc = (s_logits.argmax(1) == labels).float().mean()
        losses_m.update(loss_dict['total'].item(), imgs.size(0))
        accs_m.update(acc.item(), imgs.size(0))
        pbar.set_postfix({'loss': f"{losses_m.avg:.4f}", 'acc': f"{accs_m.avg:.4f}"})

    return losses_m.avg, accs_m.avg

@torch.no_grad()
def validate(model, loader, config):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels, _ in tqdm(loader, desc="Val"):
        imgs = imgs.to(config.DEVICE)
        logits = model(imgs)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(labels)
    acc = (torch.cat(all_preds) == torch.cat(all_labels)).float().mean().item()
    return acc

def main():
    config = Config()
    setup_seed(config.SEED)
    logger = setup_logging(config.LOG_DIR)
    logger.info("Enhanced Training Start")

    # 数据
    train_ds = NoisyDataset(config.TRAIN_DIR, transform=get_train_transform(config.IMG_SIZE, 'warmup'))
    train_size = int(0.9 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_sub, val_sub = torch.utils.data.random_split(train_ds, [train_size, val_size],
                                                       generator=torch.Generator().manual_seed(config.SEED))
    val_sub.dataset = NoisyDataset(config.TRAIN_DIR, get_val_transform(config.IMG_SIZE))

    train_loader = DataLoader(train_sub, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
                              prefetch_factor=config.PREFETCH_FACTOR, persistent_workers=config.PERSISTENT_WORKERS)
    val_loader = DataLoader(val_sub, batch_size=config.BATCH_SIZE*2, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    # 模型
    model = CLIPEnhanced(config).to(config.DEVICE)
    frozen_clip = None
    if config.LOSS_FEATURE_WEIGHT > 0:
        import open_clip
        frozen_clip, _, _ = open_clip.create_model_and_transforms(config.MODEL_NAME, pretrained='openai',
                                                                   cache_dir=config.CLIP_CACHE_DIR)
        frozen_clip.to(config.DEVICE).eval()
        for p in frozen_clip.parameters(): p.requires_grad = False

    ema_teacher = EMAModel(model, decay=config.EMA_DECAY).to(config.DEVICE) if config.USE_EMA else None

    gmm_filter = GMMFilter(len(train_ds), warmup_epochs=config.GMM_WARMUP_EPOCHS)
    class_weights = compute_class_weights(train_ds, config.NUM_CLASSES, beta=config.CLASS_BALANCE_BETA).to(config.DEVICE)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, betas=config.BETAS)
    total_steps = len(train_loader) // config.ACCUMULATION_STEPS * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=config.MIN_LR)

    criterion = CombinedLoss(config).to(config.DEVICE)
    scaler = GradScaler('cuda') if config.USE_AMP else None

    best_acc = 0.0
    patience = 0

    for epoch in range(1, config.NUM_EPOCHS+1):
        logger.info(f"--- Epoch {epoch} ---")
        train_loss, train_acc = train_one_epoch(epoch, model, ema_teacher, train_loader,
                                                criterion, optimizer, scaler, scheduler,
                                                gmm_filter, class_weights, config, logger,
                                                frozen_clip)
        if config.USE_GMM_FILTER:
            gmm_filter.fit_gmm(epoch)
            logger.info(f"GMM clean ratio: {(gmm_filter.clean_probs>0.5).float().mean():.2%}")

        val_acc = validate(model, val_loader, config)
        ema_val_acc = validate(ema_teacher.teacher, val_loader, config) if ema_teacher else 0.0
        curr_acc = max(val_acc, ema_val_acc)
        logger.info(f"Val Acc: {val_acc:.4f}, EMA Acc: {ema_val_acc:.4f}")

        if curr_acc > best_acc:
            best_acc = curr_acc
            patience = 0
            save_model = ema_teacher.teacher if ema_val_acc > val_acc and ema_teacher else model
            save_checkpoint(save_model, optimizer, scaler, epoch, best_acc, ema_teacher,
                            Path(config.CHECKPOINT_DIR)/"best_enhanced.pt")
            logger.info(f"Best model saved ({best_acc:.4f})")
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stop at epoch {epoch}")
                break

    logger.info(f"Training done. Best Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()