import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from patch_util import PatchGenerator 
from model.builder import build_model
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import json

# === Score 统一接口 ===
def get_score(model_name, model, x, output, patch_size):
    if model_name == 'autoencoder':
        return ((x - output) ** 2).mean(dim=(1, 2)).cpu().numpy()
    
    elif model_name == "pca":
        patch_score = torch.tensor(output).unsqueeze(1).repeat(1, patch_size)
        return patch_score.numpy()

    elif model_name == "iforest":
        patch_score = torch.tensor(output).unsqueeze(1).repeat(1, patch_size)
        return patch_score.numpy()   
    
    elif model_name == 'usad':
        x_hat1 = output['x_hat1']
        x_hat21 = output['x_hat21']
        mse1 = ((x - x_hat1) ** 2).mean(dim=(1, 2))
        mse21 = ((x - x_hat21) ** 2).mean(dim=(1, 2))
        return (0.5 * mse1 + 0.5 * mse21).cpu().numpy()

    elif model_name == 'anomalytrans':
        return output.cpu().numpy()

    elif model_name == 'patchtst_ae':
        return ((x - output) ** 2).mean(dim=(1, 2)).cpu().numpy()
    
    elif model_name == 'omnianomaly':
        x_hat = output['x_hat']
        kl = output['kl']
        mse = ((x - x_hat) ** 2).mean(dim=(1, 2))
        return (mse + kl).cpu().numpy()
    
    elif model_name.lower() == "tranad":
        x_hat_f = output['x_hat_f']
        x_hat_b = output['x_hat_b']
        err_f = ((x - x_hat_f) ** 2).mean(dim=(1, 2))
        err_b = ((x - x_hat_b) ** 2).mean(dim=(1, 2))
        score = (err_f + err_b) / 2
        return score.cpu().numpy()
    
    elif model_name == "dcdetector":
        series, _ = output
        if isinstance(series, list):
            # 多尺度：每个尺度取全局均值，再平均
            scores = [s.mean(dim=(1, 2, 3)) for s in series]  # 每个 [B]
            score = torch.stack(scores, dim=1).mean(dim=1)    # [B]
            patch_score = score.unsqueeze(1).repeat(1, patch_size)  # [B, P]
        else:
            # 单尺度：[B, P, 1, 1] → 去除最后两维再 squeeze
            score = series.mean(dim=(2, 3))  # [B, P]
            patch_score = score  # 直接作为 patch 分数，不 repeat
        return patch_score.cpu().numpy()
    
    elif model_name == 'npsr':
        if isinstance(output, dict):
            x_hat = output["x_hat"]  # [B, P, D]
            kl = output["kl"]        # [B]
            # 使用最大值替代均值聚合
            mse = ((x - x_hat) ** 2).mean(dim=2).max(dim=1).values  # [B]
            return (mse + kl).cpu().numpy()
        else:
            raise ValueError("NPSR 模型必须使用 return_feature=True 返回 dict 格式")
        
    else:
        raise NotImplementedError(f"打分方法未定义：{model_name}")


def point_adjust(pred, label):
    adjusted_pred = np.zeros_like(pred)
    anomaly_state = False
    for i in range(len(label)):
        if label[i] == 1 and pred[i] == 1:
            anomaly_state = True
        if anomaly_state:
            adjusted_pred[i] = 1
        if label[i] == 0:
            anomaly_state = False
    return adjusted_pred


def evaluate(scores, labels, threshold):

    pred = (scores > threshold).astype(int)
    adj_pred = point_adjust(pred, labels)
    return {
        'Precision': precision_score(labels, adj_pred),
        'Recall': recall_score(labels, adj_pred),
        'F1': f1_score(labels, adj_pred),
        'AUC': roc_auc_score(labels, scores),
        'Threshold': threshold,
        'Pred': adj_pred
    }



def plot_scores(scores, labels, threshold, pred=None, save_path=None):
    plt.figure(figsize=(14, 4))
    plt.plot(scores, label='Anomaly Score', linewidth=1.2, color='steelblue')
    plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
    plt.fill_between(np.arange(len(labels)), 0, 1, where=labels == 1,
                     color='gray', alpha=0.15,
                     transform=plt.gca().get_xaxis_transform(),
                     label='Ground Truth')

    if pred is not None:
        detected_idx = np.where(pred == 1)[0]
        plt.scatter(detected_idx, [threshold * 1.05] * len(detected_idx),
                    color='orange', marker='x', s=20, label='Detected')

    ymax = max(np.max(scores), threshold) * 1.2
    plt.ylim(0, ymax)
    plt.legend(loc='upper right')
    plt.title("Anomaly Score with Prediction and Ground Truth")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"📊 图像已保存至：{save_path}")
    else:
        plt.show()
    plt.close()


# === 主函数 ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='dataset')
    parser.add_argument('--machine', type=str, default='machine-1-1')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[16])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model', type=str, default='autoencoder')
    parser.add_argument('--patch_mode', type=str, default='sliding')
    args = parser.parse_args()

    data_dir = os.path.join(args.root_dir, 'clean_data', 'SMD')
    train_data = pd.read_csv(os.path.join(data_dir, 'train', f"{args.machine}.csv"), header=None).values
    test_data = pd.read_csv(os.path.join(data_dir, 'test', f"{args.machine}.csv"), header=None).values
    label_data = pd.read_csv(os.path.join(data_dir, 'label', f"{args.machine}.csv"), header=None).values.squeeze()

    patcher = PatchGenerator(patch_size=args.patch_size, mode=args.patch_mode)
    train_patches = patcher.generate(train_data)
    test_patches = patcher.generate(test_data)

    train_loader = DataLoader(torch.tensor(train_patches, dtype=torch.float32), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(torch.tensor(test_patches, dtype=torch.float32), batch_size=args.batch_size, shuffle=False)

    model_config = {
        "model": args.model,
        "patch_sizes": [args.patch_size],
        "input_dim": train_data.shape[1],
        "hidden_dim": 64,
        "latent_dim": 16
    }
    ckpt_path = f"checkpoints/{args.model}__{args.machine}.pth"

    if args.model in ['pca', 'iforest']:
        model = joblib.load(ckpt_path)
    else:
        model = build_model(model_config)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        model.to(args.device)
        model.eval()

    # === 阈值计算 ===
    train_scores = []
    with torch.no_grad():
        for x in train_loader:
            if args.model in ['pca', 'iforest']:
                x_np = x.numpy()
                output = model.predict(x_np)
                s = get_score(args.model, model, x, output, patch_size=args.patch_size[0])

            else:
                x = x.to(args.device)
                output = model(x, return_feature=True)
                s = get_score(args.model, model, x, output, patch_size=args.patch_size[0])
            train_scores.extend(s)
    train_scores = np.array(train_scores)
    threshold = np.percentile(train_scores, 95)

    # === 测试集 ===
    scores_patch = []
    with torch.no_grad():
        for x in test_loader:
            if args.model in ['pca', 'iforest']:
                x_np = x.numpy()
                output = model.predict(x_np)
                s = get_score(args.model, model, x, output, patch_size=args.patch_size[0])
            

            else:
                x = x.to(args.device)
                output = model(x, return_feature=True)
                s = get_score(args.model, model, x, output, patch_size=args.patch_size[0])
            scores_patch.extend(s)
    scores_patch = np.array(scores_patch)

    # === Patch → 点级 Score 对齐 ===
    if args.patch_mode.lower() == "none":
        point_score = scores_patch
    else:
        T = len(label_data)
        point_score = np.zeros(T)
        point_count = np.zeros(T)
        for i, s in enumerate(scores_patch):
            point_score[i: i + args.patch_size[0]] += s
            point_count[i: i + args.patch_size[0]] += 1
        point_score /= (point_count + 1e-8)

    # === 评估与可视化 ===
    result = evaluate(point_score, label_data, threshold)
    for k, v in result.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    
    # ✅ 保存 JSON
    result_save_path = os.path.join("vis", f"{args.model}__{args.machine}_result.json")
    serializable_result = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in result.items()}
    with open(result_save_path, "w") as f:
        json.dump(serializable_result, f, indent=2)
        
    save_path = os.path.join("vis", f"{args.model}__{args.machine}_score.png")
    plot_scores(point_score, label_data, result['Threshold'], result['Pred'], save_path=save_path)


if __name__ == '__main__':
    main()
