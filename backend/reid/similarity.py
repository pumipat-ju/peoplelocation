"""Pure vector helpers used by Re-ID and gallery code."""

import numpy as np


def l2_normalize(vec):
    if vec is None:
        return None
    array = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm < 1e-8:
        return array
    return (array / norm).astype(np.float32)


def cosine_similarity(a, b):
    if a is None or b is None:
        return -1.0
    left = l2_normalize(a)
    right = l2_normalize(b)
    return float(np.dot(left, right))
