"""
配置文件 - 增强版 (补充缺失属性)
"""
import torch
from pathlib import Path

class Config:
    # 路径
    TRAIN_DIR = r"d:\AIC-123\dataset\train"
    TEST_DIR = r"d:\AIC-123\dataset\test"
    CHECKPOINT_DIR = r"d:\AIC-123\checkpoints"
    LOG_DIR = r"d:\AIC-123\logs"
    CLIP_CACHE_DIR = r"d:\AIC-123\clip_cache"

    MODEL_NAME = "ViT-B-32"
    NUM_CLASSES = 500
    IMG_SIZE = 224

    # LoRA
    USE_LORA = True
    LORA_RANK = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["c_fc", "c_proj", "in_proj", "out_proj"]

    # Visual Prompt
    USE_VISUAL_PROMPT = True
    NUM_VISUAL_PROMPTS = 16
    PROMPT_DIM = 768

    # GMM 动态噪声过滤
    USE_GMM_FILTER = True
    GMM_WARMUP_EPOCHS = 5
    GMM_UPDATE_FREQ = 1

    # 训练参数
    NUM_EPOCHS = 30
    BATCH_SIZE = 16
    ACCUMULATION_STEPS = 8
    LEARNING_RATE = 1.5e-4
    WEIGHT_DECAY = 0.05
    BETAS = (0.9, 0.98)
    MAX_GRAD_NORM = 1.0
    WARMUP_RATIO = 0.05
    MIN_LR = 1e-6

    # 数据增强
    MIXUP_ALPHA = 0.6
    CUTMIX_ALPHA = 1.2
    MIXUP_PROB = 0.6

    # 损失函数权重
    LOSS_CE_WEIGHT = 0.4
    LOSS_DISTILL_WEIGHT = 0.4
    LOSS_FEATURE_WEIGHT = 0.0
    DISTILL_TEMPERATURE = 4.0
    USE_SYMMETRIC_CE = True
    SYMMETRIC_ALPHA = 0.5
    SYMMETRIC_BETA = 1.5
    # 以下三个是 CombinedLoss 必需的
    BITEMPERED_T1 = 0.8
    BITEMPERED_T2 = 2.0
    USE_DISTILLATION = True   # 总开关

    # EMA 教师
    USE_EMA = True
    EMA_DECAY = 0.9995

    # 伪标签
    USE_PSEUDO_LABEL = True
    PSEUDO_LABEL_START_EPOCH = 10
    PSEUDO_LABEL_UPDATE_FREQ = 3
    PSEUDO_LABEL_THRESHOLD = 0.85

    # 类别平衡
    CLASS_BALANCE_BETA = 0.95

    # 优化
    USE_AMP = True
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2
    PERSISTENT_WORKERS = True

    # 早停
    EARLY_STOPPING_PATIENCE = 8

    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG_INTERVAL = 100
    SAVE_INTERVAL = 5

    def __init__(self):
        for d in [self.CHECKPOINT_DIR, self.LOG_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)