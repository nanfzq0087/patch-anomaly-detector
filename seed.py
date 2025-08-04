# utils/seed.py
import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多卡

    # 确保 CUDA 算子确定性（性能略低，但结果可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
