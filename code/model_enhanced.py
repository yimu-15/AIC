"""
增强模型 - Visual Prompt (安全版) + Attention LoRA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from pathlib import Path

class LoRALayer(nn.Module):
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
    def __init__(self, linear, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha, dropout)

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

class InProjWithLoRA(nn.Module):
    def __init__(self, in_proj_weight, in_proj_bias, embed_dim, rank=16, alpha=32, dropout=0.0):
        super().__init__()
        self.register_buffer('in_proj_weight', in_proj_weight)
        self.register_buffer('in_proj_bias', in_proj_bias)
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(embed_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, 3 * embed_dim))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        lora_out = self.dropout(x @ self.lora_A @ self.lora_B) * self.scaling
        return qkv + lora_out

class VisualPromptWrapper(nn.Module):
    def __init__(self, clip_visual, num_prompts=16, prompt_dim=768):
        super().__init__()
        self.visual = clip_visual
        self.num_prompts = num_prompts
        self.prompts = nn.Parameter(torch.zeros(num_prompts, prompt_dim))
        nn.init.normal_(self.prompts, std=0.02)

    def forward(self, x):
        visual = self.visual
        # 1. patch embedding
        x = visual.conv1(x)           # [B, dim, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, N, dim]

        # 2. class token
        class_token = visual.class_embedding.expand(x.shape[0], 1, -1)  # [B, 1, dim]
        x = torch.cat([class_token, x], dim=1)  # [B, 1+N, dim]

        # 3. 插入 prompts（class token 之后，patch tokens 之前）
        prompts = self.prompts.expand(x.shape[0], -1, -1)  # [B, num_prompts, dim]
        x = torch.cat([x[:, :1, :], prompts, x[:, 1:, :]], dim=1)  # [B, 1+num_prompts+N, dim]

        # 4. 位置编码扩展（关键修复）
        pos_embed = visual.positional_embedding  # [1, 1+N, dim] or [1+N, dim]
        if pos_embed.dim() == 2:
            pos_embed = pos_embed.unsqueeze(0)    # [1, 1+N, dim]

        # 创建新的位置编码，形状 [1, 1+num_prompts+N, dim]
        new_pos = torch.zeros(1, 1 + self.num_prompts + pos_embed.size(1) - 1, pos_embed.size(2), device=x.device)
        new_pos[:, 0:1, :] = pos_embed[:, 0:1, :]                     # class token 位置不变
        new_pos[:, 1+self.num_prompts:, :] = pos_embed[:, 1:, :]      # patch tokens 位置后移
        # prompts 位置保持零（或其他可学习的初始化）

        x = x + new_pos

        # 5. transformer
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)   # [seq_len, B, dim]
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)   # [B, seq_len, dim]
        x = visual.ln_post(x[:, 0, :])  # class token

        if visual.proj is not None:
            x = x @ visual.proj
        return x / x.norm(dim=-1, keepdim=True)

class CLIPEnhanced(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        cache_dir = Path(config.CLIP_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            config.MODEL_NAME, pretrained='openai', cache_dir=str(cache_dir)
        )
        self.tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)

        for p in self.clip_model.parameters():
            p.requires_grad = False

        if config.USE_LORA:
            self._inject_lora()

        self.use_visual_prompt = config.USE_VISUAL_PROMPT
        if self.use_visual_prompt:
            self.visual_prompt = VisualPromptWrapper(
                self.clip_model.visual,
                num_prompts=config.NUM_VISUAL_PROMPTS,
                prompt_dim=config.PROMPT_DIM
            )
        else:
            self.visual_prompt = None

        self.feature_dim = self.clip_model.visual.output_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, config.NUM_CLASSES)
        )
        nn.init.normal_(self.classifier[2].weight, std=0.01)
        nn.init.zeros_(self.classifier[2].bias)

    def _inject_lora(self):
        target = self.config.LORA_TARGET_MODULES
        rank = self.config.LORA_RANK
        alpha = self.config.LORA_ALPHA
        dropout = self.config.LORA_DROPOUT
        modified = 0

        for blk in self.clip_model.visual.transformer.resblocks:
            if 'c_fc' in target and hasattr(blk.mlp, 'c_fc'):
                blk.mlp.c_fc = LinearWithLoRA(blk.mlp.c_fc, rank, alpha, dropout)
                modified += 1
            if 'c_proj' in target and hasattr(blk.mlp, 'c_proj'):
                blk.mlp.c_proj = LinearWithLoRA(blk.mlp.c_proj, rank, alpha, dropout)
                modified += 1

            if hasattr(blk, 'attn'):
                if 'in_proj' in target and hasattr(blk.attn, 'in_proj_weight'):
                    weight = blk.attn.in_proj_weight.data.clone()
                    bias = blk.attn.in_proj_bias.data.clone()
                    embed_dim = blk.attn.embed_dim
                    blk.attn.in_proj = InProjWithLoRA(weight, bias, embed_dim, rank, alpha, dropout)
                    modified += 1
                if 'out_proj' in target and hasattr(blk.attn, 'out_proj'):
                    blk.attn.out_proj = LinearWithLoRA(blk.attn.out_proj, rank, alpha, dropout)
                    modified += 1

        print(f"✓ Enhanced LoRA injected into {modified} layers")

    def encode_image(self, images):
        if self.use_visual_prompt:
            return self.visual_prompt(images)
        feats = self.clip_model.encode_image(images)
        return feats / feats.norm(dim=-1, keepdim=True)

    def forward(self, images, return_features=False):
        feats = self.encode_image(images)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits