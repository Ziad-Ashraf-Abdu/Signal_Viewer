# backend/shared/downsample.py
import numpy as np

def downsample_without_anti_aliasing(data, orig_sr, target_fs):
    if target_fs >= orig_sr:
        return data, orig_sr
    ratio = orig_sr / target_fs
    new_len = int(len(data) / ratio)
    indices = (np.arange(new_len) * ratio).astype(int)
    indices = indices[indices < len(data)]
    aliased = data[indices]
    return aliased, target_fs
