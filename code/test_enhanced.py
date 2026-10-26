"""
测试推理脚本 - 生成 pred_results.csv
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from pathlib import Path

from config_enhanced import Config
from dataset import TestDataset, get_test_transform
from model_enhanced import CLIPEnhanced
from utils import load_checkpoint

def main():
    config = Config()
    device = config.DEVICE

    # 加载模型
    model = CLIPEnhanced(config).to(device)
    checkpoint_path = Path(config.CHECKPOINT_DIR) / "best_enhanced.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    # 加载时不需要 optimizer 等
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    # 测试数据集
    test_dataset = TestDataset(config.TEST_DIR, transform=get_test_transform(config.IMG_SIZE))
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE*2, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)

    results = []
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            for name, pred in zip(names, preds):
                results.append([name, f"{pred:04d}"])

    df = pd.DataFrame(results, columns=['filename', 'category'])
    output_path = "pred_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(results)} predictions to {output_path}")

if __name__ == "__main__":
    main()