import subprocess
from tqdm import tqdm
import argparse
import json
import glob
import numpy as np
import pandas as pd

# ✅ 支持从命令行传入 patch_mode
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='autoencoder')
parser.add_argument('--patch_size', type=int, nargs='+', default=[16],
                    help='支持多个 patch size，比如 --patch_size 3 5 7')
parser.add_argument('--patch_mode', type=str, default='sliding', help='Patch 切分模式')
parser.add_argument('--root_dir', type=str, default='dataset')
args = parser.parse_args()

# 🧠 拼接 patch_size 字符串（解决 subprocess 报错问题）
patch_size_str = " ".join(str(p) for p in args.patch_size)

# === 所有机器列表 ===
machine_ids = [
    "machine-1-1",#"machine-1-2","machine-1-3", "machine-1-4","machine-1-5","machine-1-6", "machine-1-7","machine-1-8",
    #"machine-2-1","machine-2-2","machine-2-3", "machine-2-4","machine-2-5","machine-2-6", "machine-2-7","machine-2-8","machine-2-9",
    #"machine-3-1","machine-3-2","machine-3-3", "machine-3-4","machine-3-5","machine-3-6", "machine-3-7","machine-3-8","machine-3-9","machine-3-10","machine-3-11"
]

# === 批量训练 ===
print(f"\n=== 🔧 批量训练模型（{args.model}） ===")
for machine in tqdm(machine_ids):
    cmd = (
        f"python3 main.py "
        f"--machine {machine} "
        f"--patch_size {patch_size_str} "
        f"--root_dir {args.root_dir} "
        f"--model {args.model} "
        f"--patch_mode {args.patch_mode}"
    )
    subprocess.run(cmd, shell=True, check=True)

# === 批量测试 ===
print(f"\n=== 🧪 批量测试模型（{args.model}） ===")
for machine in tqdm(machine_ids):
    cmd = (
        f"python3 inference.py "
        f"--machine {machine} "
        f"--patch_size {patch_size_str} "
        f"--root_dir {args.root_dir} "
        f"--model {args.model} "
        f"--patch_mode {args.patch_mode}"
    )
    subprocess.run(cmd, shell=True, check=True)

# === 汇总 JSON 结果 ===
print(f"\n=== 📊 汇总模型指标（macro average） ===")
result_files = glob.glob(f"vis/{args.model}__machine-*_result.json")

all_metrics = {"Machine": [], "Precision": [], "Recall": [], "F1": [], "AUC": []}
for file in result_files:
    with open(file, "r") as f:
        res = json.load(f)
        machine_name = file.split("__")[-1].replace("_result.json", "")
        all_metrics["Machine"].append(machine_name)
        for k in ["Precision", "Recall", "F1", "AUC"]:
            all_metrics[k].append(res.get(k, np.nan))

# === 打印平均指标 ===
for k in ["Precision", "Recall", "F1", "AUC"]:
    scores = all_metrics[k]
    avg = np.nanmean(scores)
    print(f"{k} (avg): {avg:.4f}")

# ✅ 导出 CSV
df = pd.DataFrame(all_metrics)
csv_path = f"vis/{args.model}_all_results.csv"
df.to_csv(csv_path, index=False)
print(f"\n✅ 所有机器指标已保存至：{csv_path}")
