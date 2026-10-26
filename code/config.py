"""
配置文件 — 鲁棒修复版（500 类噪声标签）
========================================
针对 500 类 + 噪声标签场景优化默认值。
"""
import torch
from pathlib import Path


class Config:
    # ==================== 路径 ====================
    TRAIN_DIR = r"d:\AIC-123\dataset\train"
    TEST_DIR = r"d:\AIC-123\dataset\test"
    CHECKPOINT_DIR = r"d:\AIC-123\checkpoints"
    LOG_DIR = r"d:\AIC-123\logs"
    CLIP_CACHE_DIR = r"d:\AIC-123\clip_cache"

    # ==================== 模型 ====================
    MODEL_NAME = "ViT-B-32"
    NUM_CLASSES = 500
    IMG_SIZE = 224

    # ==================== LoRA ====================
    USE_LORA = True
    LORA_RANK = 16           # ↑ 从 12 提升到 16（更多适配容量）
    LORA_ALPHA = 32          # ↑ 从 24 提升到 32（更强的 LoRA 信号）
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["c_fc", "c_proj", "q_proj", "k_proj", "v_proj", "out_proj"]

    # Attention LoRA
    USE_ATTENTION_LORA = True

    # ==================== VPT ====================
    USE_VPT = True
    VPT_NUM_TOKENS = 16      # ↑ 从 10 提升到 16
    VPT_DROPOUT = 0.1

    # ==================== 训练 ====================
    NUM_EPOCHS = 30
    BATCH_SIZE = 16
    ACCUMULATION_STEPS = 4   # ↓ 从 8 降到 4（有效 batch=64，更快更新）

    LEARNING_RATE = 5.0e-4   # ↑ 从 2e-4 提升（配合 xavier 分类头）
    WEIGHT_DECAY = 0.01      # ↓ 从 0.05 降低（减少正则化强度）
    BETAS = (0.9, 0.98)
    MAX_GRAD_NORM = 1.0

    # 学习率调度
    WARMUP_RATIO = 0.05
    MIN_LR = 1e-6

    # ==================== 动态损失权重 ====================
    # Stage 1 (epoch 1-5):   CE-heavy (CE=0.7, distill=0.2, feature=0.1)
    # Stage 2 (epoch 6-12):  balanced  (CE→0.5, distill→0.35, feature→0.15)
    # Stage 3 (epoch 13+):   distill-heavy (CE=0.3, distill=0.55, feature=0.15)
    DYNAMIC_WEIGHT_START_EPOCH = 6   # Stage 1→2 切换点
    DYNAMIC_WEIGHT_MID_EPOCH = 13    # Stage 2→3 切换点

    # ==================== 损失函数静态默认值 ====================
    # （会被动态权重覆盖，这里设置最终阶段的默认值）
    LOSS_CE_WEIGHT = 0.3
    LOSS_DISTILL_WEIGHT = 0.55
    LOSS_FEATURE_WEIGHT = 0.15

    DISTILL_TEMPERATURE = 4.0

    BITEMPERED_T1 = 0.8
    BITEMPERED_T2 = 2.0

    USE_SYMMETRIC_CE = True
    SYMMETRIC_ALPHA = 0.5
    SYMMETRIC_BETA = 1.0    # ↓ 从 1.5 降低（RCE 权重不应超过 CE）

    # ==================== 数据增强 ====================
    MIXUP_ALPHA = 0.6
    CUTMIX_ALPHA = 1.2
    MIXUP_PROB = 0.6

    RANDAUGMENT_N = 2
    RANDAUGMENT_M = 10

    # ==================== Trust Filter ====================
    TRUST_FILTER_START_EPOCH = 5
    TRUST_UPDATE_FREQUENCY = 4
    TRUST_MOMENTUM = 0.9

    LOSS_HISTORY_LENGTH = 5
    CONFIDENCE_THRESHOLD = 0.75

    # ==================== GMM 动态噪声评分 ====================
    USE_GMM_SCORING = True
    GMM_START_EPOCH = 5
    GMM_UPDATE_FREQUENCY = 4

    # ==================== CLIP 原型零样本评分 ====================
    USE_CLIP_PROTOTYPE_SCORING = True

    # ==================== 三路评分融合 ====================
    FUSION_W1 = 0.4    # trust_scores 语义一致性 EMA
    FUSION_W2 = 0.3    # gmm_clean_prob 动态噪声概率
    FUSION_W3 = 0.3    # clip_proto_scores CLIP 原型静态评分

    # ==================== 特征正则化 ====================
    USE_FEATURE_REG = True
    LOSS_FEATURE_REG_WEIGHT = 0.05

    # ==================== EMA ====================
    USE_EMA = True
    EMA_DECAY = 0.9995

    # ==================== Co-teaching ====================
    USE_COTEACHING = False
    COTEACHING_EPOCHS = 8
    COTEACHING_FORGET_RATE = 0.25
    COTEACHING_GRADUAL = True

    # ==================== 伪标签 ====================
    USE_PSEUDO_LABEL = True
    PSEUDO_LABEL_START_EPOCH = 10
    PSEUDO_LABEL_UPDATE_FREQ = 3
    PSEUDO_LABEL_THRESHOLD = 0.85

    # ==================== 类别平衡 ====================
    CLASS_BALANCE_BETA = 0.95

    # ==================== 优化 ====================
    USE_AMP = True
    USE_DISTILLATION = True

    NUM_WORKERS = 4
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2
    PERSISTENT_WORKERS = True

    # ==================== 早停 ====================
    EARLY_STOPPING_PATIENCE = 8

    # ==================== 其他 ====================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOG_INTERVAL = 100
    SAVE_INTERVAL = 5

    def __init__(self):
        for dir_path in [self.CHECKPOINT_DIR, self.LOG_DIR]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
