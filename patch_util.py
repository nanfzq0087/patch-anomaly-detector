import numpy as np

class PatchGenerator:
    def __init__(self, patch_size, mode='sliding', stride=1):
        if isinstance(patch_size, int):
            self.patch_sizes = [patch_size]
        else:
            self.patch_sizes = patch_size
        self.mode = mode
        self.stride = stride

    def generate(self, data):
        """
        :param data: [T, D]
        :return: [N, patch_size, D]
        """
        if self.mode == "sliding":
            return self._sliding(data)
        elif self.mode == "non-overlap":
            return self._non_overlap(data)
        elif self.mode == "bidirectional":
            return self._bidirectional(data)
        elif self.mode == "none":
            return self._none(data)
        else:
            raise ValueError(f"Unsupported patching mode: {self.mode}")

    def _sliding(self, data):
        patches = []
        T, D = data.shape
        for p in self.patch_sizes:
            for i in range(0, T - p + 1, self.stride):
                patch = data[i:i+p]
                patches.append(patch)
        return np.stack(patches)  # [N, p, D]

    def _non_overlap(self, data):
        T, D = data.shape
        patches = []
        for i in range(0, T - self.patch_size + 1, self.patch_size):
            patch = data[i:i + self.patch_size]
            patches.append(patch)
        return np.stack(patches)

    def _bidirectional(self, data):
        T, D = data.shape
        fwd_patches = []
        bwd_patches = []
        for i in range(0, T - self.patch_size + 1, self.stride):
            fwd = data[i:i + self.patch_size]
            bwd = data[i + self.patch_size - 1:i - 1:-1]  # 反向切 patch
            if bwd.shape[0] == self.patch_size:
                fwd_patches.append(fwd)
                bwd_patches.append(bwd[::-1])  # 再反转回来
        return np.stack(fwd_patches + bwd_patches)
    
    def _none(self, data):
        """
        不做切片，直接返回 [1, T, D]，作为单个 patch
        """
        return np.expand_dims(data, axis=0)  # [1, T, D]
    
# ✅ 原有的 magnitude 计算保持不变
def compute_magnitude(patches):
    """
    :param patches: [N, patch_size, D]
    :return: [N,] 每个 patch 的 L2 范数
    """
    return np.linalg.norm(patches, axis=(1, 2))
