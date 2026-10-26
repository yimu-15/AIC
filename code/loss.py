"""
损失函数 — 鲁棒修复版
=====================
支持:
  - 动态权重覆盖（ce_weight / distill_weight / feature_weight / feature_reg_weight）
  - 对称交叉熵（SCE）
  - 知识蒸馏（KL 散度，支持温度缩放）
  - 特征对齐（MSE）
  - 特征正则化（锚定冻结 CLIP）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """组合损失函数（支持动态权重覆盖）"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 默认权重（可被 forward 的动态参数覆盖）
        self.ce_weight = config.LOSS_CE_WEIGHT
        self.distill_weight = config.LOSS_DISTILL_WEIGHT
        self.feature_weight = config.LOSS_FEATURE_WEIGHT
        self.feature_reg_weight = getattr(config, 'LOSS_FEATURE_REG_WEIGHT', 0.0)

        self.temperature = config.DISTILL_TEMPERATURE
        self.use_symmetric_ce = config.USE_SYMMETRIC_CE
        self.symmetric_alpha = config.SYMMETRIC_ALPHA
        self.symmetric_beta = config.SYMMETRIC_BETA

    def forward(self, student_logits, teacher_logits, targets,
                lam=None, targets_a=None, targets_b=None,
                sample_weights=None, student_features=None, teacher_features=None,
                reference_features=None,
                # ── 动态权重覆盖（可选） ──
                ce_weight=None, distill_weight=None,
                feature_weight=None, feature_reg_weight=None):
        """
        Args:
            student_logits:    学生输出 [B, C]
            teacher_logits:    教师输出 [B, C]
            targets:           标签 [B]
            lam, targets_a/b:  Mixup 参数
            sample_weights:    样本权重 [B]
            student_features:  学生特征 [B, D]
            teacher_features:  教师特征 [B, D]
            reference_features: 冻结参考 CLIP 特征 [B, D]
            ce_weight etc.:    动态权重覆盖（None 时使用 config 默认值）
        """
        # 解析权重
        _ce_w = ce_weight if ce_weight is not None else self.ce_weight
        _distill_w = distill_weight if distill_weight is not None else self.distill_weight
        _feat_w = feature_weight if feature_weight is not None else self.feature_weight
        _reg_w = feature_reg_weight if feature_reg_weight is not None else self.feature_reg_weight

        losses = {}

        # ═════════════════════════════════════════════════════════
        # 1. 交叉熵损失
        # ═════════════════════════════════════════════════════════
        if lam is not None and targets_a is not None and targets_b is not None:
            # Mixup/CutMix
            ce_a = F.cross_entropy(student_logits, targets_a, reduction='none')
            ce_b = F.cross_entropy(student_logits, targets_b, reduction='none')
            ce_loss = lam * ce_a + (1 - lam) * ce_b
        else:
            if self.use_symmetric_ce:
                ce_loss = self._symmetric_cross_entropy(student_logits, targets)
            else:
                ce_loss = F.cross_entropy(student_logits, targets, reduction='none')

        if sample_weights is not None:
            ce_loss = (ce_loss * sample_weights).mean()
        else:
            ce_loss = ce_loss.mean()

        losses['ce'] = ce_loss

        # ═════════════════════════════════════════════════════════
        # 2. 知识蒸馏损失（KL 散度）
        # ═════════════════════════════════════════════════════════
        if self.config.USE_DISTILLATION:
            distill_loss = self._distillation_loss(student_logits, teacher_logits)
            losses['distill'] = distill_loss
        else:
            losses['distill'] = torch.tensor(0.0, device=student_logits.device)

        # ═════════════════════════════════════════════════════════
        # 3. 特征对齐损失（学生 ↔ 教师）
        # ═════════════════════════════════════════════════════════
        if student_features is not None and teacher_features is not None:
            # 使用余弦距离替代 MSE（对特征模长更鲁棒）
            s_n = F.normalize(student_features.float(), dim=-1)
            t_n = F.normalize(teacher_features.float(), dim=-1)
            feature_loss = (1 - (s_n * t_n).sum(dim=-1)).mean()
            losses['feature'] = feature_loss
        else:
            losses['feature'] = torch.tensor(0.0, device=student_logits.device)

        # ═════════════════════════════════════════════════════════
        # 4. 特征正则化（锚定冻结 CLIP）
        # ═════════════════════════════════════════════════════════
        if reference_features is not None:
            s_n = F.normalize(student_features.float(), dim=-1)
            r_n = F.normalize(reference_features.float(), dim=-1)
            reg_loss = (1 - (s_n * r_n).sum(dim=-1)).mean()
            losses['feature_reg'] = reg_loss
        else:
            losses['feature_reg'] = torch.tensor(0.0, device=student_logits.device)

        # ═════════════════════════════════════════════════════════
        # 总损失
        # ═════════════════════════════════════════════════════════
        total = (
            _ce_w * losses['ce']
            + _distill_w * losses['distill']
            + _feat_w * losses['feature']
            + _reg_w * losses['feature_reg']
        )

        losses['total'] = total

        # 记录实际使用的权重（调试用）
        losses['_ce_w'] = _ce_w
        losses['_distill_w'] = _distill_w
        losses['_feat_w'] = _feat_w
        losses['_reg_w'] = _reg_w

        return total, losses

    def _distillation_loss(self, student_logits, teacher_logits):
        """知识蒸馏（KL 散度 + 温度缩放）"""
        s_log_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        t_prob = F.softmax(teacher_logits / self.temperature, dim=-1)

        kl = F.kl_div(s_log_prob, t_prob, reduction='batchmean')
        return kl * (self.temperature ** 2)

    def _symmetric_cross_entropy(self, logits, targets):
        """
        对称交叉熵 (SCE):
          SCE = α · CE + β · RCE
        其中 RCE 是 Reverse Cross Entropy，对噪声标签更鲁棒
        """
        # 前向 CE
        ce = F.cross_entropy(logits, targets, reduction='none')

        # 反向 CE: -Σ t · log(p)
        probs = F.softmax(logits, dim=-1)
        target_probs = torch.zeros_like(probs).scatter_(
            1, targets.unsqueeze(1), 1.0
        )
        rce = -torch.sum(target_probs * torch.log(probs + 1e-7), dim=-1)

        return self.symmetric_alpha * ce + self.symmetric_beta * rce
