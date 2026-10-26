"""
GMM 动态噪声过滤器
"""
import numpy as np
from sklearn.mixture import GaussianMixture
import torch

class GMMFilter:
    def __init__(self, num_samples, warmup_epochs=5):
        self.num_samples = num_samples
        self.warmup_epochs = warmup_epochs
        self.clean_probs = torch.ones(num_samples)
        self.loss_buffer = torch.zeros(num_samples)
        self.loss_count = torch.zeros(num_samples)

    @torch.no_grad()
    def update(self, per_sample_losses, indices):
        self.loss_buffer[indices] += per_sample_losses.cpu()
        self.loss_count[indices] += 1

    def fit_gmm(self, epoch):
        if epoch < self.warmup_epochs:
            self.clean_probs.fill_(1.0)
            return self.clean_probs

        avg_loss = self.loss_buffer / self.loss_count.clamp(min=1)
        avg_loss[self.loss_count == 0] = avg_loss[self.loss_count > 0].median()

        losses_np = avg_loss.numpy().reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=100, random_state=0)
        gmm.fit(losses_np)
        prob = gmm.predict_proba(losses_np)
        clean_component = gmm.means_.argmin()
        self.clean_probs = torch.from_numpy(prob[:, clean_component]).float()

        self.loss_buffer.zero_()
        self.loss_count.zero_()
        return self.clean_probs

    def get_clean_weights(self, indices):
        return self.clean_probs[indices]