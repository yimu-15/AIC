"""
loss_enhanced.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ce_weight = config.LOSS_CE_WEIGHT
        self.distill_weight = config.LOSS_DISTILL_WEIGHT
        self.feature_weight = config.LOSS_FEATURE_WEIGHT

        self.temperature = config.DISTILL_TEMPERATURE
        self.use_symmetric_ce = config.USE_SYMMETRIC_CE
        self.symmetric_alpha = config.SYMMETRIC_ALPHA
        self.symmetric_beta = config.SYMMETRIC_BETA

    def forward(self, student_logits, teacher_logits, targets,
                lam=None, targets_a=None, targets_b=None,
                sample_weights=None, student_features=None, teacher_features=None,
                feature_reg_loss=None):
        losses = {}

        if lam is not None and targets_a is not None and targets_b is not None:
            ce_loss_a = F.cross_entropy(student_logits, targets_a, reduction="none")
            ce_loss_b = F.cross_entropy(student_logits, targets_b, reduction="none")
            ce_loss = lam * ce_loss_a + (1 - lam) * ce_loss_b
        else:
            if self.use_symmetric_ce:
                ce_loss = self.symmetric_cross_entropy(student_logits, targets)
            else:
                ce_loss = F.cross_entropy(student_logits, targets, reduction="none")

        if sample_weights is not None:
            ce_loss = (ce_loss * sample_weights).mean()
        else:
            ce_loss = ce_loss.mean()

        losses["ce"] = ce_loss

        if self.config.USE_DISTILLATION:
            distill_loss = self.distillation_loss(student_logits, teacher_logits)
        else:
            distill_loss = torch.tensor(0.0, device=student_logits.device)
        losses["distill"] = distill_loss

        if feature_reg_loss is not None:
            losses["feature"] = feature_reg_loss
        elif student_features is not None and teacher_features is not None:
            losses["feature"] = F.mse_loss(student_features, teacher_features)
        else:
            losses["feature"] = torch.tensor(0.0, device=student_logits.device)

        total = (
            self.ce_weight * losses["ce"] +
            self.distill_weight * losses["distill"] +
            self.feature_weight * losses["feature"]
        )
        losses["total"] = total

        return total, losses

    def distillation_loss(self, student_logits, teacher_logits):
        sp = F.log_softmax(student_logits / self.temperature, dim=-1)
        tp = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl = F.kl_div(sp, tp, reduction="batchmean")
        return kl * (self.temperature ** 2)

    def symmetric_cross_entropy(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        probs = F.softmax(logits, dim=-1)
        target_probs = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
        rce = -torch.sum(target_probs * torch.log(probs + 1e-7), dim=-1)
        return self.symmetric_alpha * ce + self.symmetric_beta * rce
