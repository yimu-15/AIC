"""
工具函数 — 鲁棒修复版
=====================
新增:
  - log_gradient_norms: 按参数组打印梯度范数（调试用）
  - 修复 LabelSmoothingCrossEntropy 缺少 F 导入
"""
import os
import random
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math


def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir):
    """设置日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "train.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def save_checkpoint(model, optimizer, scaler, epoch, acc, ema_model, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'acc': acc,
    }

    if ema_model is not None:
        checkpoint['ema_model_state_dict'] = ema_model.teacher.state_dict()

    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, scaler, ema_model, checkpoint_path):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scaler is not None and checkpoint['scaler_state_dict'] is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    if ema_model is not None and 'ema_model_state_dict' in checkpoint:
        ema_model.teacher.load_state_dict(checkpoint['ema_model_state_dict'])

    epoch = checkpoint['epoch']
    acc = checkpoint['acc']

    print(f"✓ Checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {epoch}, Acc: {acc:.4f}")

    return epoch, acc


def load_pretrained_weights(model, checkpoint_path, device='cpu'):
    """
    宽松加载（strict=False）：热启动专用，不要求模型结构完全一致。
    只加载权重，不恢复 optimizer/scaler/epoch 状态。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)

    print(f"✓ Warm-started weights from {checkpoint_path}")
    print(f"  Missing keys (fresh init): {len(missing_keys)}")
    if missing_keys:
        print(f"    e.g. {missing_keys[:5]}")
    print(f"  Unexpected keys (ignored): {len(unexpected_keys)}")
    if unexpected_keys:
        print(f"    e.g. {unexpected_keys[:5]}")

    if 'acc' in checkpoint:
        print(f"  Checkpoint's recorded val acc: {checkpoint['acc']:.4f}")

    return missing_keys, unexpected_keys


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EMAModel(nn.Module):
    """EMA 模型包装器"""
    def __init__(self, model, decay=0.999):
        super().__init__()
        self.teacher = self._create_ema_model(model)
        self.decay = decay
        self.num_updates = 0

    def _create_ema_model(self, model):
        import copy
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for ema_param, model_param in zip(self.teacher.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)

    def forward(self, *args, **kwargs):
        return self.teacher(*args, **kwargs)


def compute_class_weights(dataset, num_classes, beta=0.9999):
    """计算类别平衡权重（Class-Balanced Loss）"""
    class_counts = torch.zeros(num_classes)

    for _, label, _ in dataset:
        class_counts[label] += 1

    effective_num = 1.0 - torch.pow(beta, class_counts)
    weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-8)
    weights = weights / weights.sum() * num_classes

    return weights


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    num_cycles=0.5, min_lr=0.0):
    """Cosine 学习率调度（带 warmup）"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

        return max(min_lr / optimizer.defaults['lr'], cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


def mixup_data(x, y, alpha=1.0):
    """Mixup 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    _, _, h, w = x.size()
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    y_a, y_b = y, y[index]

    return x, y_a, y_b, lam


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵"""
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size(-1)
        log_preds = F.log_softmax(output, dim=-1)

        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()

        return loss * self.epsilon / c + (1 - self.epsilon) * F.nll_loss(
            log_preds, target, reduction=self.reduction
        )


def compute_kl_div(p_logits, q_logits, temperature=1.0):
    """计算 KL 散度"""
    p = F.softmax(p_logits / temperature, dim=-1)
    log_q = F.log_softmax(q_logits / temperature, dim=-1)
    kl = F.kl_div(log_q, p, reduction='batchmean') * (temperature ** 2)
    return kl


def accuracy(output, target, topk=(1,)):
    """计算 Top-K 准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


@torch.no_grad()
def update_bn_stats(model, dataloader, device, num_batches=100):
    """更新 BN 统计量（用于 EMA 模型）"""
    model.train()
    for batch_idx, (images, _, _) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        images = images.to(device)
        _ = model(images)
    model.eval()


def get_learning_rate(optimizer):
    """获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def count_parameters(model):
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


def format_time(seconds):
    """格式化时间"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ═══════════════════════════════════════════════════════════════════
# 梯度监控（调试用）
# ═══════════════════════════════════════════════════════════════════

def log_gradient_norms(model, logger, prefix=""):
    """
    打印各参数组的梯度范数，用于调试梯度消失/爆炸。

    分组:
      - classifier:  分类头参数
      - lora:        LoRA A/B 参数
      - vpt:         VPT prompt embeddings
      - other:       其他可训练参数
    """
    groups = {'classifier': [], 'lora': [], 'vpt': [], 'other': []}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        gn = param.grad.norm().item()

        if 'classifier' in name:
            groups['classifier'].append((name, gn))
        elif 'lora' in name.lower():
            groups['lora'].append((name, gn))
        elif 'prompt' in name.lower():
            groups['vpt'].append((name, gn))
        else:
            groups['other'].append((name, gn))

    lines = [f"[GradNorm] {prefix}"]
    for gname, entries in groups.items():
        if not entries:
            continue
        gn_mean = np.mean([e[1] for e in entries])
        gn_max = np.max([e[1] for e in entries])
        gn_min = np.min([e[1] for e in entries])
        lines.append(f"  {gname}: mean={gn_mean:.6f} max={gn_max:.6f} min={gn_min:.6f} "
                     f"({len(entries)} params)")

    # 总梯度
    all_gn = []
    for entries in groups.values():
        all_gn.extend([e[1] for e in entries])
    if all_gn:
        total_gn = np.sqrt(sum(g**2 for g in all_gn))
        lines.append(f"  TOTAL: {total_gn:.6f}")

    logger.info("\n".join(lines))


def visualize_predictions(images, labels, preds, class_names=None, num_images=8):
    """可视化预测结果（调试用）"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx in range(min(num_images, len(images))):
        img = images[idx].cpu().permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225])
               + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)

        axes[idx].imshow(img)

        true_label = labels[idx].item()
        pred_label = preds[idx].item()

        if class_names is not None:
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
            title = f"True: {true_name}\nPred: {pred_name}"
        else:
            title = f"True: {true_label}\nPred: {pred_label}"

        color = 'green' if true_label == pred_label else 'red'
        axes[idx].set_title(title, color=color)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def save_metrics(metrics, save_path):
    """保存训练指标"""
    import json
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved to {save_path}")


def load_metrics(load_path):
    """加载训练指标"""
    import json
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    return metrics
