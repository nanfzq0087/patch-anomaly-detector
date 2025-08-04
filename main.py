import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from model.builder import build_model
from train import train_model
from patch_util import PatchGenerator, compute_magnitude
import matplotlib.pyplot as plt
from seed import set_seed


def load_csv_series(data_dir, machine_id):
    path = os.path.join(data_dir, f"{machine_id}.csv")
    df = pd.read_csv(path, header=None)
    return df.values  # shape: [T, D]

def normalize_data(train, test):
    mean = train.mean(axis=0)         # [D]
    std = train.std(axis=0) + 1e-8    # 防止除0
    train_norm = (train - mean) / std
    test_norm = (test - mean) / std
    return train_norm, test_norm

def prepare_datasets(root_dir, machine_id, patch_size,  patch_mode, plot=True):
    data_dir = os.path.join(root_dir, 'clean_data', 'SMD')
    print("📥 读取原始 CSV 数据...")
    train_data = load_csv_series(os.path.join(data_dir, 'train'), machine_id)  # [T, D]
    test_data = load_csv_series(os.path.join(data_dir, 'test'), machine_id)    # [T, D]
    test_label = load_csv_series(os.path.join(data_dir, 'label'), machine_id)  # [T,]

    # ✅ 按列标准化
    train_data, test_data = normalize_data(train_data, test_data)

    if plot:
        # 🎨 可视化归一化后的训练数据前几个特征
        num_plot_dims = min(5, train_data.shape[1])
        plt.figure(figsize=(12, 6))
        for i in range(num_plot_dims):
            plt.plot(train_data[:, i], label=f"Dim {i}")
        plt.title(f"Normalized Train Data for {machine_id}")
        plt.xlabel("Time")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.tight_layout()
        os.makedirs("vis", exist_ok=True)
        plt.savefig(f"vis/{machine_id}_train_preview.png")
        print(f"📊 训练数据图已保存至 vis/{machine_id}_train_preview.png")
        plt.close()

    print(f"🔄 执行 Patch 切分...（模式: {patch_mode}）")
    patcher = PatchGenerator(patch_size=patch_size, mode=patch_mode)
    train_x = patcher.generate(train_data)
    test_x = patcher.generate(test_data)

    train_mag = compute_magnitude(train_x)
    test_mag = compute_magnitude(test_x)

    train_set = TensorDataset(torch.tensor(train_x, dtype=torch.float))
    test_set = TensorDataset(torch.tensor(test_x, dtype=torch.float))
    return train_set, test_set

def main():
    set_seed(43)  # 你自己设定的种子值
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='dataset')
    parser.add_argument('--machine', type=str, default='machine-1-1')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[16],
                    help='支持多个 patch size，比如 --patch_size 3 5 7')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model', type=str, default='my_model', help='模型名称，例如 my_model、autoencoder、usad')
    parser.add_argument('--patch_mode', type=str, default='sliding',
                    help='Patch 切分模式')
    parser.add_argument('--beta', type=float, default=1.0, help='KL loss 权重系数（用于 NPSR）')
    args = parser.parse_args()

    train_set, test_set = prepare_datasets(args.root_dir, args.machine, args.patch_size, args.patch_mode)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    print("✅ 初始化模型")
    model_config = {
        "model": args.model,
        "patch_sizes": [args.patch_size],
        "input_dim": 38,
        "hidden_dim": 64,
        "latent_dim": 16,
    }
    model = build_model(model_config)
    if isinstance(model, torch.nn.Module):
        model.to(args.device)

    print("🎯 开始训练")
    train_model(model, train_loader, test_loader, args, args.machine)

if __name__ == '__main__':
    main()