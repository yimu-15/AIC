"""
模型定义 — 鲁棒修复版
=====================
修复要点:
  1. encode_image 显式传 normalize=False，避免双归一化
  2. VPT wrapper 完全对齐 open_clip VisionTransformer.forward 逻辑
  3. MultiheadAttentionLoRA 基于实际张量形状处理 batch_first
  4. 分类头改用 xavier_uniform（适配 500 类）
  5. 增加特征统计日志，便于梯度调试
  6. GMM + CLIP 原型 + Trust EMA 三路评分融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from pathlib import Path
import copy
import numpy as np
from tqdm import tqdm

try:
    from sklearn.mixture import GaussianMixture
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
# LoRA 基础组件
# ═══════════════════════════════════════════════════════════════════

class LoRALayer(nn.Module):
    """LoRA 低秩适配层"""
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.dropout(x @ self.lora_A @ self.lora_B) * self.scaling


class LinearWithLoRA(nn.Module):
    """带 LoRA 的线性层（MLP c_fc / c_proj）"""
    def __init__(self, linear, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features,
            rank=rank, alpha=alpha, dropout=dropout
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features


# ═══════════════════════════════════════════════════════════════════
# Attention LoRA（鲁棒版）
# ═══════════════════════════════════════════════════════════════════

class MultiheadAttentionLoRA(nn.Module):
    """
    包装 nn.MultiheadAttention，为 Q/K/V/Out 投影注入 LoRA。

    关键修复:
      - 从原始 forward 的实际张量形状推断 batch_first，而非依赖属性
      - 完全复现 PyTorch MultiheadAttention 的计算流程（含 in_proj 拆分、
        scaled_dot_product_attention、out_proj）
      - Q/K/V LoRA 增量作用在投影后的 (L,N,E) 空间上
    """
    def __init__(self, orig_attn, rank=8, alpha=16, dropout=0.0,
                 target_modules=("q_proj", "k_proj", "v_proj", "out_proj")):
        super().__init__()
        if not isinstance(orig_attn, nn.MultiheadAttention):
            raise TypeError(
                f"MultiheadAttentionLoRA only wraps nn.MultiheadAttention, "
                f"got {type(orig_attn).__name__}"
            )

        self.orig_attn = orig_attn
        for p in self.orig_attn.parameters():
            p.requires_grad = False

        self.embed_dim = orig_attn.embed_dim
        self.num_heads = orig_attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        # ⚠ 旧属性可能不可靠，forward 时从张量形状推断
        self._batch_first_attr = getattr(orig_attn, "batch_first", False)

        def _make_lora():
            return LoRALayer(self.embed_dim, self.embed_dim,
                             rank=rank, alpha=alpha, dropout=dropout)

        self.lora_q = _make_lora() if "q_proj" in target_modules else None
        self.lora_k = _make_lora() if "k_proj" in target_modules else None
        self.lora_v = _make_lora() if "v_proj" in target_modules else None
        self.lora_out = _make_lora() if "out_proj" in target_modules else None

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None, **kwargs):
        # ── 推断 batch_first ──
        # 若 query.shape = (B, L, E)，则 batch_first=True
        # 若 query.shape = (L, B, E)，则 batch_first=False
        is_batch_first = (query.dim() == 3 and query.shape[0] == key.shape[0]
                          and query.shape[1] != key.shape[1])

        if is_batch_first:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))

        L, N, E = query.shape
        S = key.shape[0]

        # ── Q/K/V 投影 ──
        in_proj_weight = self.orig_attn.in_proj_weight  # [3E, E]
        in_proj_bias = self.orig_attn.in_proj_bias        # [3E] or None

        w_q, w_k, w_v = in_proj_weight.chunk(3, dim=0)
        if in_proj_bias is not None:
            b_q, b_k, b_v = in_proj_bias.chunk(3, dim=0)
        else:
            b_q = b_k = b_v = None

        q = F.linear(query, w_q, b_q)
        k = F.linear(key,   w_k, b_k)
        v = F.linear(value, w_v, b_v)

        # LoRA 增量
        if self.lora_q is not None:
            q = q + self.lora_q(query)
        if self.lora_k is not None:
            k = k + self.lora_k(key)
        if self.lora_v is not None:
            v = v + self.lora_v(value)

        # ── reshape → attention → reshape ──
        q = q.contiguous().view(L, N * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(S, N * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(S, N * self.num_heads, self.head_dim).transpose(0, 1)

        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0
        )

        attn_output = attn_output.transpose(0, 1).contiguous().view(L, N, E)

        # ── 输出投影 ──
        out = self.orig_attn.out_proj(attn_output)
        if self.lora_out is not None:
            out = out + self.lora_out(attn_output)

        if is_batch_first:
            out = out.transpose(0, 1)

        return out, None


# ═══════════════════════════════════════════════════════════════════
# Visual Prompt Tuning（鲁棒版）
# ═══════════════════════════════════════════════════════════════════

class VPTVisualWrapper(nn.Module):
    """
    VPT-Shallow: 在 class token 之后、patch tokens 之前插入可学习 prompt。

    修复要点:
      1. 严格对齐 open_clip VisionTransformer.forward 的张量操作
      2. dtype/device 全部从输入推断，不硬编码
      3. positional_embedding 扩展与冻结部分精确对齐
      4. 自检: 若 open_clip 结构不兼容则自动降级为直通
    """
    REQUIRED_ATTRS = [
        "conv1", "class_embedding", "positional_embedding",
        "ln_pre", "transformer", "ln_post"
    ]

    def __init__(self, visual, num_tokens=10, dropout=0.0):
        super().__init__()
        self.visual = visual
        self.num_tokens = num_tokens
        self.enabled = all(hasattr(visual, a) for a in self.REQUIRED_ATTRS)

        if not self.enabled:
            missing = [a for a in self.REQUIRED_ATTRS if not hasattr(visual, a)]
            print(f"⚠ VPT disabled: missing {missing}")
            return

        width = visual.class_embedding.shape[0]
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, width))
        nn.init.trunc_normal_(self.prompt_embeddings, std=0.02)
        self.prompt_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 冻结原始 visual 所有参数
        for p in self.visual.parameters():
            p.requires_grad = False

        print(f"✓ VPT: {num_tokens} prompts, width={width}")

    @property
    def output_dim(self):
        return self.visual.output_dim

    def forward(self, x):
        if not self.enabled:
            return self.visual(x)

        v = self.visual
        dtype = x.dtype
        device = x.device
        B = x.shape[0]

        # ── 1. Patch embedding（完全对齐 open_clip） ──
        x = v.conv1(x)                                    # [B, C, H, W]
        x = x.reshape(B, x.shape[1], -1)                  # [B, C, N]
        x = x.permute(0, 2, 1)                            # [B, N, C]

        # ── 2. Class token ──
        cls_token = (v.class_embedding.to(dtype=dtype)
                     .unsqueeze(0).unsqueeze(0)
                     .expand(B, 1, -1))                   # [B, 1, C]
        x = torch.cat([cls_token, x], dim=1)              # [B, 1+N, C]

        # ── 3. 插入 prompt tokens ──
        prompt = (self.prompt_dropout(self.prompt_embeddings)
                  .to(dtype=dtype)
                  .expand(B, -1, -1))                      # [B, P, C]
        x = torch.cat([x[:, :1, :], prompt, x[:, 1:, :]], dim=1)  # [B, 1+P+N, C]

        # ── 4. 位置编码扩展 ──
        pos_embed = v.positional_embedding.to(dtype=dtype)  # [1+N, C] or [1, 1+N, C]
        if pos_embed.dim() == 3:
            pos_embed = pos_embed.squeeze(0)                 # → [1+N, C]

        P = self.num_tokens
        N_patch = pos_embed.shape[0] - 1                     # patch 数量
        C_pos = pos_embed.shape[1]

        new_pos = torch.zeros(1 + P + N_patch, C_pos,
                              dtype=dtype, device=device)
        new_pos[0:1, :] = pos_embed[0:1, :]                  # cls
        new_pos[1+P:, :] = pos_embed[1:, :]                  # patches
        # prompt 位置用零初始化（避免污染预训练位置先验）

        x = x + new_pos.unsqueeze(0)                          # broadcast [B,...]

        # ── 5. Transformer（完全对齐 open_clip） ──
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)                                # NLD → LND
        x = v.transformer(x)
        x = x.permute(1, 0, 2)                                # LND → NLD

        # ── 6. 池化 + 投影 ──
        x = v.ln_post(x[:, 0, :])                              # class token only

        if getattr(v, "proj", None) is not None:
            x = x @ v.proj.to(dtype=dtype)

        return x


# ═══════════════════════════════════════════════════════════════════
# CLIP + LoRA + VPT + 分类头
# ═══════════════════════════════════════════════════════════════════

class CLIPWithLoRA(nn.Module):
    """CLIP ViT + LoRA + VPT + 分类头"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ── 加载预训练 CLIP ──
        cache_dir = Path(config.CLIP_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading CLIP: {config.MODEL_NAME} (pretrained=openai)")
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            config.MODEL_NAME,
            pretrained='openai',
            cache_dir=str(cache_dir)
        )

        # ── 冻结 CLIP 主干 ──
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # ── 注入 LoRA ──
        if config.USE_LORA:
            self._inject_lora()

        # ── VPT（必须在 LoRA 之后，确保 visual 已被包装的属性可被 LoRA 找到） ──
        # 注意：VPT 会替换 self.clip_model.visual，但 LoRA 已经注入了 transformer 内部的层
        # 所以 VPT wrapper 内的 transformer 仍然包含 LoRA 参数
        self.feature_dim = self.clip_model.visual.output_dim

        if getattr(config, "USE_VPT", False):
            self.clip_model.visual = VPTVisualWrapper(
                self.clip_model.visual,
                num_tokens=config.VPT_NUM_TOKENS,
                dropout=config.VPT_DROPOUT
            )

        # ── 分类头 ──
        # 修复：使用 xavier_uniform 初始化，适配 500 类别
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, config.NUM_CLASSES)
        )
        self._init_classifier()

    def _init_classifier(self):
        """改进的分类头初始化 —— 为 500 类提供更强的初始梯度"""
        linear = self.classifier[-1]
        # xavier_uniform 比 N(0, 0.01) 更适合高维输出空间
        nn.init.xavier_uniform_(linear.weight, gain=0.5)
        nn.init.zeros_(linear.bias)
        print(f"✓ Classifier init: xavier_uniform(gain=0.5) for {self.config.NUM_CLASSES} classes")

    def _inject_lora(self):
        """注入 LoRA 到 MLP + Attention"""
        target_modules = self.config.LORA_TARGET_MODULES
        rank = self.config.LORA_RANK
        alpha = self.config.LORA_ALPHA
        dropout = self.config.LORA_DROPOUT

        mlp_count = 0
        attn_count = 0

        attn_targets = [m for m in ("q_proj", "k_proj", "v_proj", "out_proj")
                        if m in target_modules]
        use_attn_lora = (getattr(self.config, "USE_ATTENTION_LORA", False)
                         and len(attn_targets) > 0)

        for block_idx, block in enumerate(self.clip_model.visual.transformer.resblocks):
            # MLP LoRA
            if 'c_fc' in target_modules and hasattr(block.mlp, 'c_fc'):
                block.mlp.c_fc = LinearWithLoRA(
                    block.mlp.c_fc, rank=rank, alpha=alpha, dropout=dropout
                )
                mlp_count += 1

            if 'c_proj' in target_modules and hasattr(block.mlp, 'c_proj'):
                block.mlp.c_proj = LinearWithLoRA(
                    block.mlp.c_proj, rank=rank, alpha=alpha, dropout=dropout
                )
                mlp_count += 1

            # Attention LoRA
            if use_attn_lora and isinstance(block.attn, nn.MultiheadAttention):
                block.attn = MultiheadAttentionLoRA(
                    block.attn, rank=rank, alpha=alpha, dropout=dropout,
                    target_modules=attn_targets
                )
                attn_count += 1

        print(f"✓ LoRA injected: {mlp_count} MLP + {attn_count} Attention blocks "
              f"(rank={rank}, alpha={alpha})")

        if mlp_count == 0 and attn_count == 0:
            print("⚠ WARNING: No LoRA layers injected! Check LORA_TARGET_MODULES.")
            if len(self.clip_model.visual.transformer.resblocks) > 0:
                b = self.clip_model.visual.transformer.resblocks[0]
                print(f"  MLP children: {[n for n, _ in b.mlp.named_children()]}")
                print(f"  Attn type: {type(b.attn).__name__}")

    def forward(self, images, return_features=False):
        """
        前向传播（修复版）:
          - 显式调用 visual(images)，避免 encode_image normalize 不确定性
          - 单次 L2 归一化
        """
        # 直接使用 visual（已被 VPT wrapper 替换，或原始 visual）
        image_features = self.clip_model.visual(images)

        # 单次 L2 归一化
        image_features = F.normalize(image_features.float(), dim=-1)

        # 分类
        logits = self.classifier(image_features)

        if return_features:
            return logits, image_features
        return logits


# ═══════════════════════════════════════════════════════════════════
# Trust Filter（GMM + CLIP 原型 + 语义一致性 三路融合）
# ═══════════════════════════════════════════════════════════════════

class TrustFilter:
    """
    三路噪声过滤评分:
      1. trust_scores   — 语义一致性 EMA（类内特征相似度）
      2. gmm_clean_prob — GMM 动态噪声概率（基于 loss_history 双高斯拟合）
      3. clip_proto_scores — CLIP 图像原型静态零样本评分（训练全程不变）

    融合: fused = W1*trust + W2*gmm + W3*proto
    """

    def __init__(self, num_samples, num_classes, config):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.config = config

        # (1) 语义一致性 EMA
        self.trust_scores = torch.ones(num_samples)

        # (2) Loss history 环形缓冲区 → GMM
        buf_len = config.LOSS_HISTORY_LENGTH
        self.loss_history = torch.zeros(num_samples, buf_len)
        self.loss_ptr = 0
        self.loss_valid_epochs = 0

        # GMM 干净概率（未拟合前视为全部干净）
        self.gmm_clean_prob = torch.ones(num_samples)

        # (3) CLIP 原型静态评分
        self.clip_proto_scores = torch.full((num_samples,), 0.5)

        # 伪标签
        self.pseudo_labels = torch.full((num_samples,), -1, dtype=torch.long)
        self.pseudo_confidences = torch.zeros(num_samples)

        # 特征统计标志（调试用）
        self._feature_stats_logged = False

    # ── 语义一致性评分 ──
    @torch.no_grad()
    def compute_trust_scores(self, model, dataloader, device, config):
        """计算类内特征相似度作为语义一致性评分"""
        model.eval()

        all_features = []
        all_labels = []
        all_indices = []

        for images, labels, indices in dataloader:
            images = images.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', enabled=config.USE_AMP):
                _, features = model(images, return_features=True)

            all_features.append(features.cpu().float())
            all_labels.append(labels)
            all_indices.append(indices)

        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)
        all_indices = torch.cat(all_indices)

        # 调试: 打印特征统计
        if not self._feature_stats_logged:
            fnorm = all_features.norm(dim=-1)
            print(f"  [Trust] Feature stats: norm mean={fnorm.mean():.4f} "
                  f"std={fnorm.std():.4f} min={fnorm.min():.4f} max={fnorm.max():.4f}")
            self._feature_stats_logged = True

        all_features = F.normalize(all_features.float(), dim=-1)

        trust_scores = torch.zeros(self.num_samples)

        for class_id in range(self.num_classes):
            mask = all_labels == class_id
            if mask.sum() <= 1:
                continue

            class_features = all_features[mask]
            class_indices = all_indices[mask]

            # 类内成对余弦相似度均值
            sim_matrix = class_features @ class_features.T
            sim_matrix.fill_diagonal_(0)
            mean_sim = sim_matrix.sum(dim=1) / max(mask.sum() - 1, 1)
            trust_scores[class_indices] = mean_sim

        # 归一化到 [0, 1]
        t_min, t_max = trust_scores.min(), trust_scores.max()
        if t_max - t_min > 1e-8:
            trust_scores = (trust_scores - t_min) / (t_max - t_min)

        # EMA 更新
        momentum = config.TRUST_MOMENTUM
        self.trust_scores = (momentum * self.trust_scores
                             + (1 - momentum) * trust_scores)

        clean_ratio = (self.trust_scores > config.CONFIDENCE_THRESHOLD).float().mean()
        print(f"  Trust scores updated | clean_ratio={clean_ratio:.2%} | "
              f"mean={self.trust_scores.mean():.4f}")

    # ── 损失历史管理 ──
    def update_loss_history(self, indices, losses):
        self.loss_history[indices.cpu(), self.loss_ptr] = losses.detach().cpu().float()

    def advance_epoch(self):
        self.loss_valid_epochs = min(
            self.loss_valid_epochs + 1,
            self.config.LOSS_HISTORY_LENGTH
        )
        self.loss_ptr = (self.loss_ptr + 1) % self.config.LOSS_HISTORY_LENGTH

    # ── GMM 动态噪声评分 ──
    def fit_gmm(self):
        """基于样本历史平均损失拟合双高斯混合模型"""
        if not _SKLEARN_AVAILABLE:
            print("  [GMM] sklearn not available, skipping")
            return

        valid_cols = min(self.loss_valid_epochs, self.config.LOSS_HISTORY_LENGTH)
        if valid_cols == 0:
            print("  [GMM] No loss history yet, skipping")
            return

        avg_loss = self.loss_history[:, :valid_cols].mean(dim=1).numpy()

        # 过滤从未参与训练的样本
        has_loss = (self.loss_history[:, :valid_cols].sum(dim=1).numpy() != 0)
        num_valid = int(has_loss.sum())
        if num_valid < 10:
            print(f"  [GMM] Only {num_valid} samples have loss history, skipping")
            return

        losses_for_fit = avg_loss[has_loss].reshape(-1, 1)

        # 归一化提升数值稳定性
        l_min, l_max = losses_for_fit.min(), losses_for_fit.max()
        if l_max - l_min < 1e-8:
            print("  [GMM] Loss variance too small, skipping")
            return
        losses_norm = (losses_for_fit - l_min) / (l_max - l_min)

        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-3,
                              reg_covar=1e-6, random_state=42)
        gmm.fit(losses_norm)

        # 低损失成分 = 干净样本
        clean_component = int(np.argmin(gmm.means_.flatten()))
        probs = gmm.predict_proba(losses_norm)
        clean_probs = probs[:, clean_component]

        # 未参与拟合的样本给中性值 0.5
        full_clean = np.full(self.num_samples, 0.5, dtype=np.float32)
        full_clean[np.where(has_loss)[0]] = clean_probs

        self.gmm_clean_prob = torch.from_numpy(full_clean).float()

        clean_ratio = (self.gmm_clean_prob[has_loss] > 0.5).float().mean()
        print(f"  [GMM] Fitted on {num_valid} samples | "
              f"estimated clean ratio = {clean_ratio:.2%} | "
              f"means = ({gmm.means_[0,0]:.4f}, {gmm.means_[1,0]:.4f})")

    # ── CLIP 原型零样本评分 ──
    def set_clip_proto_scores(self, scores):
        """设置静态 CLIP 图像原型评分（训练全程不变）"""
        self.clip_proto_scores = scores.detach().float().cpu()
        print(f"  ✓ CLIP prototype scores loaded "
              f"(mean={self.clip_proto_scores.mean():.4f}, "
              f"std={self.clip_proto_scores.std():.4f})")

    # ── 三路评分融合 ──
    def get_fused_scores(self, indices):
        """
        fused = W1*trust + W2*gmm + W3*proto
        三路信号覆盖:
          - trust: 特征空间语义一致性（动态 EMA）
          - gmm:   损失空间噪声概率（动态 GMM）
          - proto: CLIP 图像原型零样本评分（静态锚点）
        """
        w1 = self.config.FUSION_W1
        w2 = self.config.FUSION_W2
        w3 = self.config.FUSION_W3

        trust = self.trust_scores[indices]
        gmm = self.gmm_clean_prob[indices]
        proto = self.clip_proto_scores[indices]

        fused = w1 * trust + w2 * gmm + w3 * proto
        return fused

    def get_sample_weights(self, indices, class_weights, labels):
        """组合：融合评分 × 类别平衡权重"""
        fused = self.get_fused_scores(indices).to(labels.device)
        class_w = class_weights[labels]
        weights = fused * class_w
        weights = weights / (weights.mean() + 1e-8)
        return weights

    # ── 伪标签 ──
    @torch.no_grad()
    def update_pseudo_labels(self, model, dataloader, device, threshold=0.8):
        model.eval()
        corrected = 0
        total = 0

        for images, labels, indices in dataloader:
            images = images.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', enabled=model.training):
                logits = model(images)

            probs = F.softmax(logits, dim=-1)
            max_probs, preds = probs.max(dim=1)

            high_conf = max_probs.cpu() > threshold
            high_trust = self.trust_scores[indices] > 0.7
            valid = high_conf & high_trust

            if valid.sum() > 0:
                v_idx = indices[valid]
                self.pseudo_labels[v_idx] = preds[valid].cpu()
                self.pseudo_confidences[v_idx] = max_probs[valid].cpu()
                corrected += valid.sum().item()
            total += len(indices)

        print(f"  Pseudo labels: {corrected}/{total} ({corrected/total:.2%}) corrected")

    def get_corrected_labels(self, indices, original_labels):
        corrected = original_labels.clone()
        for i, idx in enumerate(indices):
            if self.pseudo_labels[idx] != -1:
                corrected[i] = self.pseudo_labels[idx]
        return corrected


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_clip_prototype_scores(config, dataset):
    """
    使用独立冻结的 OpenCLIP 计算每类图像特征原型，
    以样本与自身类别原型的余弦相似度作为"零样本"评分。
    该评分在训练全程保持不变。
    """
    from dataset import get_val_transform
    from torch.utils.data import DataLoader

    print("\n[Prototype Scoring] Loading independent frozen CLIP...")
    cache_dir = Path(config.CLIP_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    proto_clip, _, _ = open_clip.create_model_and_transforms(
        config.MODEL_NAME,
        pretrained='openai',
        cache_dir=str(cache_dir)
    )
    proto_clip = proto_clip.to(config.DEVICE)
    proto_clip.eval()
    for p in proto_clip.parameters():
        p.requires_grad = False

    # 使用确定性 transform
    scoring_dataset = copy.copy(dataset)
    scoring_dataset.transform = get_val_transform(config.IMG_SIZE)

    loader = DataLoader(
        scoring_dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    all_features, all_labels, all_indices = [], [], []

    for images, labels, indices in tqdm(loader, desc="  Extracting features"):
        images = images.to(config.DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', enabled=config.USE_AMP):
            features = proto_clip.encode_image(images, normalize=True)

        all_features.append(features.cpu().float())
        all_labels.append(labels)
        all_indices.append(indices)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    all_indices = torch.cat(all_indices)

    # 类别原型
    feature_dim = all_features.size(1)
    prototypes = torch.zeros(config.NUM_CLASSES, feature_dim)
    for c in range(config.NUM_CLASSES):
        mask = all_labels == c
        if mask.sum() > 0:
            prototypes[c] = F.normalize(all_features[mask].mean(dim=0), dim=0)

    # 相似度评分
    sample_protos = prototypes[all_labels]
    sims = (all_features.float() * sample_protos).sum(dim=1)

    # 归一化到 [0, 1]
    s_min, s_max = sims.min(), sims.max()
    if s_max - s_min > 1e-8:
        sims_norm = (sims - s_min) / (s_max - s_min)
    else:
        sims_norm = sims

    scores = torch.full((len(dataset),), 0.5)
    scores[all_indices] = sims_norm

    del proto_clip
    torch.cuda.empty_cache()

    print(f"  ✓ Prototype scores: mean={sims_norm.mean():.4f} std={sims_norm.std():.4f}")
    return scores


def load_frozen_reference_clip(config):
    """
    加载独立冻结的 CLIP 模型作为特征正则化锚点。
    该模型常驻显存，不参与梯度更新。
    """
    cache_dir = Path(config.CLIP_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Feature Reg] Loading frozen reference CLIP...")
    ref_clip, _, _ = open_clip.create_model_and_transforms(
        config.MODEL_NAME,
        pretrained='openai',
        cache_dir=str(cache_dir)
    )
    ref_clip.eval()
    for p in ref_clip.parameters():
        p.requires_grad = False

    print("  ✓ Reference CLIP loaded (frozen anchor)")
    return ref_clip


class EMATeacher(nn.Module):
    """EMA 教师模型"""

    def __init__(self, student_model, decay=0.999):
        super().__init__()
        self.teacher = copy.deepcopy(student_model)
        self.decay = decay
        self.num_updates = 0
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, student_model):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for tp, sp in zip(self.teacher.parameters(), student_model.parameters()):
            tp.data.mul_(decay).add_(sp.data, alpha=1 - decay)

    def forward(self, *args, **kwargs):
        return self.teacher(*args, **kwargs)


def create_model(config, num_samples):
    """工厂函数：创建模型 + EMA 教师 + Trust Filter"""
    print("\n" + "=" * 50)
    print("Creating model...")
    print("=" * 50)

    model = CLIPWithLoRA(config)

    ema_teacher = None
    if config.USE_EMA:
        ema_teacher = EMATeacher(model, decay=config.EMA_DECAY)
        print(f"✓ EMA Teacher created (decay={config.EMA_DECAY})")

    trust_filter = TrustFilter(num_samples, config.NUM_CLASSES, config)
    print(f"✓ TrustFilter initialized (3-way fusion: GMM + CLIP proto + trust EMA)")

    return model, ema_teacher, trust_filter
