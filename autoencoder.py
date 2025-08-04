# model/autoencoder.py
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=38, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.patch_generator = None  # ← 新增

    def set_patch_generator(self, patch_generator):
        """设置 patch generator 对象"""
        self.patch_generator = patch_generator

    def apply_patch(self, x):
        """
        将输入的原始数据 [T, D] 切成 patch [N, P, D]
        x: torch.Tensor [T, D]
        return: torch.Tensor [N, P, D]
        """
        assert self.patch_generator is not None, "Patch generator 未设置"
        patches = self.patch_generator(x.cpu().numpy())  # [N, P, D]
        return torch.tensor(patches, dtype=torch.float, device=x.device)

    def forward(self, x, return_feature=False):
        """
        支持 [B, P, D] 输入 或原始数据 [T, D] 自动切片
        增加 return_feature 参数，用于推理时返回 feature
        """
        if x.dim() == 2:  # [T, D]，需要切 patch
            x = self.apply_patch(x)  # [N, P, D]

        if x.dim() == 3:
            B, P, D = x.shape
            x = x.view(B * P, D)
            z = self.encoder(x)
            x_hat = self.decoder(z).view(B, P, D)
        else:
            z = self.encoder(x)
            x_hat = self.decoder(z)

        if return_feature:
            return x_hat  # AE 不用额外 feature，直接返回重构值
        else:
            return x_hat

