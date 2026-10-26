"""
utils_enhanced.py
"""
import os
import random
import logging
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
import open_clip


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(log_file=None):
    logger = logging.getLogger("enhanced_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


@torch.no_grad()
def build_zeroshot_classifier(clip_model, class_names, device, templates=None):
    if templates is None:
        templates = [
            "a photo of category {}.",
            "a photo of a {}.",
            "an image of category {}.",
            "a close-up photo of category {}.",
            "a blurry photo of category {}."
        ]

    weights = []
    for cls in class_names:
        texts = [t.format(cls) for t in templates]
        tokens = open_clip.tokenize(texts).to(device)
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        class_feat = text_features.mean(dim=0)
        class_feat = class_feat / class_feat.norm()
        weights.append(class_feat)

    zeroshot_weights = torch.stack(weights, dim=1)  # [D, C]
    return zeroshot_weights


@torch.no_grad()
def compute_clip_trust_scores(clip_model, dataloader, zeroshot_weights, device, temperature=100.0):
    clip_model.eval()
    all_scores = []
    all_indices = []

    for images, labels, indices in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        feats = clip_model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        logits = temperature * feats @ zeroshot_weights
        probs = logits.softmax(dim=-1)

        scores = probs[torch.arange(len(labels), device=device), labels]
        all_scores.append(scores.detach().cpu())
        all_indices.append(indices)

    return torch.cat(all_scores), torch.cat(all_indices)


def gmm_split(losses):
    losses = np.array(losses).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=100, random_state=0)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses)
    clean_component = gmm.means_.argmin()
    return prob[:, clean_component]


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
