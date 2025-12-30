# src/volume_integral.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import cv2


def load_mask_bool(mask_path: str | Path) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    return (m > 0)


def relative_volume_integral(depth_ref: np.ndarray, depth_tgt: np.ndarray, mask_bool: np.ndarray) -> float:
    """
    Ölçeksiz (relative) hacim: mask içindeki |ref - tgt| toplamı.
    Birimi yoktur, calib ile ölçeklenir.
    """
    if depth_ref.shape != depth_tgt.shape:
        raise ValueError("depth_ref and depth_tgt shape mismatch")
    if mask_bool.shape != depth_tgt.shape:
        raise ValueError("mask shape mismatch")

    diff = np.abs(depth_ref - depth_tgt).astype(np.float32)
    vals = diff[mask_bool]
    if vals.size < 10:
        raise RuntimeError("Mask too small for volume estimation")
    return float(np.sum(vals))


def estimate_volume_m3(
    depth_ref: np.ndarray,
    depth_calib: np.ndarray,
    mask_calib: np.ndarray,
    calib_volume_m3: float,
    depth_test: np.ndarray,
    mask_test: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Returns:
      volume_test_m3, scale, rel_calib, rel_test
    """
    rel_calib = relative_volume_integral(depth_ref, depth_calib, mask_calib)
    rel_test  = relative_volume_integral(depth_ref, depth_test,  mask_test)

    if rel_calib <= 1e-8:
        raise RuntimeError("Calibration relative volume too small")

    scale = calib_volume_m3 / rel_calib
    vol_test = rel_test * scale
    return float(vol_test), float(scale), float(rel_calib), float(rel_test)


def upsert_json(path: str | Path, key: str, payload: dict) -> None:
    path = Path(path)
    data = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    data[key] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
