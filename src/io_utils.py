# src/io_utils.py

import cv2
import numpy as np
import os
from pathlib import Path


# -------------------------------------------------
# IO HELPERS
# -------------------------------------------------
def load_image(path: Path | str):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def save_depth_npy(depth: np.ndarray, path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), depth)


def ensure_dir(path: Path | str):
    Path(path).mkdir(parents=True, exist_ok=True)
