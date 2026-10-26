import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import logging

sys.path.insert(0, 'D:/AIC-123/code')
from config import Config
from dataset import NoisyDataset, get_train_transform, get_val_transform
from model import create_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALPHA = 0.5
TEMPERATURE = 4.0
TEACHER_PT = r'D:/AIC-123/checkpoints/best_model.pt'
STUDENT_PT = r'D:/AIC-123/checkpoints/distilled_model.pt'


def distill_loss(s_logits, t_logits, labels):
    hard = F.cross_entropy(s_logits, labels)
    soft = F.kl_div(
        F.log_softmax(s_logits / TEMPERATURE, dim=1),
        F.softmax(t_logits / TEMPERATURE, dim=1),
        reduction='batchmean'
    ) * (TEMPERATURE ** 2)
    return ALPHA * hard + (1 - ALPHA) * soft


def load_teacher(config, num_samples):
    ckpt = torch.load(TEACHER_PT, map_location='cpu', weights_only=False)
    teacher, _, _ = create_model(config, num_samples)
    teacher.load_state_dict(ckpt['model_state_dict'])
    teacher.to(config.DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    logger.info(f"Teacher loaded | Epoch {ckpt['epoch']} | Acc {ckpt.get('acc', 'N/A')}")
    return teacher


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels, *_ in tqdm(loader, desc="Validating", leave=False):
        images, labels = images.to(device), labels.to(device)
        correct += model(images).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    config = Config()

    full_dataset = NoisyDataset(config.TRAIN_DIR,
                                transform=get_train_transform(config.IMG_SIZE, phase='warmup'))
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    val_subset.dataset = NoisyDataset(config.TRAIN_DIR,
                                      transform=get_val_transform(config.IMG_SIZE))

    train_loader = DataLoader(
        train_subset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.BATCH_SIZE * 2, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    logger.info(f"Train: {len(train_subset)} | Val: {len(val_subset)}")

    teacher = load_teacher(config, len(full_dataset))

    student, _, _ = create_model(config, len(full_dataset))
    student.to(config.DEVICE)
    logger.info(f"Student trainable params: {sum(p.numel() for p in student.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler('cuda', enabled=config.USE_AMP)

    best_acc, patience = 0.0, 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        student.train()
        total_loss = correct = total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}")
        for images, labels, *_ in pbar:
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=config.USE_AMP):
                s_logits = student(images)
                with torch.no_grad():
                    t_logits = teacher(images)
                loss = distill_loss(s_logits, t_logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            correct += s_logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=total_loss / (pbar.n + 1), acc=correct / total)

        val_acc = validate(student, val_loader, config.DEVICE)
        logger.info(f"Epoch {epoch} | Val Acc: {val_acc:.4f} | Best: {best_acc:.4f}")

        if val_acc > best_acc:
            best_acc, patience = val_acc, 0
            torch.save({'epoch': epoch, 'model_state_dict': student.state_dict(), 'acc': val_acc}, STUDENT_PT)
            logger.info(f"Best student saved (Acc: {val_acc:.4f})")
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping.")
                break

    logger.info(f"Done. Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()