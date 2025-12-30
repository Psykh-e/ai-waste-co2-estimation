# src/volume_pointcloud.py
from __future__ import annotations
import numpy as np


def _compute_fx_fy(W: int, H: int, fov_x_deg: float | None = None, fov_y_deg: float | None = None) -> tuple[float, float]:
    """
    Pinhole: fx = W / (2*tan(FOVx/2)), fy = H / (2*tan(FOVy/2))
    Eğer sadece fov_x verilirse fy'yi aspect'e göre türetir.
    """
    if fov_x_deg is None and fov_y_deg is None:
        raise ValueError("Provide at least one of fov_x_deg or fov_y_deg")

    if fov_x_deg is not None:
        fovx = np.deg2rad(float(fov_x_deg))
        fx = W / (2.0 * np.tan(fovx / 2.0))
    else:
        fx = None

    if fov_y_deg is not None:
        fovy = np.deg2rad(float(fov_y_deg))
        fy = H / (2.0 * np.tan(fovy / 2.0))
    else:
        # fov_y yoksa aspect oranıyla yaklaşıkla
        # tan(fovy/2) ~= (H/W) * tan(fovx/2)
        fovy = 2.0 * np.arctan((H / W) * np.tan(fovx / 2.0))
        fy = H / (2.0 * np.tan(fovy / 2.0))

    if fx is None:
        # fov_x yoksa aspect ile fx'i yaklaşıkla
        fovx = 2.0 * np.arctan((W / H) * np.tan(fovy / 2.0))
        fx = W / (2.0 * np.tan(fovx / 2.0))

    return float(fx), float(fy)


def depthdiff_to_pointcloud(
    depth_ref: np.ndarray,
    depth_tgt: np.ndarray,
    mask_bool: np.ndarray,
    fov_x_deg: float = 60.0,
    fov_y_deg: float | None = None,
    h_min: float = 1e-6,
    clip_percentile: float = 99.5,
) -> np.ndarray:
    """
    Z olarak |ref - tgt| kullanır (yükseklik).
    X,Y için pinhole normalize ray kullanır: (x-cx)/fx, (y-cy)/fy.

    Not: MiDaS ölçeği belirsiz → tüm metrik ölçek CALIB ile çözülecek.
    """
    if depth_ref.shape != depth_tgt.shape or depth_tgt.shape != mask_bool.shape:
        raise ValueError("shape mismatch")

    H, W = depth_tgt.shape
    ys, xs = np.where(mask_bool)
    if xs.size < 10:
        raise RuntimeError("Mask too small")

    h = np.abs(depth_ref[ys, xs] - depth_tgt[ys, xs]).astype(np.float32)

    # küçük/gürültü yüksekliklerini at
    keep = h > float(h_min)
    xs, ys, h = xs[keep], ys[keep], h[keep]
    if h.size < 10:
        raise RuntimeError("Not enough valid height pixels")

    # outlier kırp (convex hull şişmesini ve voxel gürültüsünü azaltır)
    hi = np.percentile(h, float(clip_percentile))
    h = np.clip(h, 0.0, hi)

    fx, fy = _compute_fx_fy(W, H, fov_x_deg=fov_x_deg, fov_y_deg=fov_y_deg)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    x_cam = (xs.astype(np.float32) - cx) / fx
    y_cam = (ys.astype(np.float32) - cy) / fy
    z_cam = h  # "yükseklik"

    pts = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)
    return pts


def relative_volume_voxel(points: np.ndarray, voxel_size: float = 0.01) -> float:
    """
    Voxel hacmi: nokta bulutunu voxel grid'e oturtup dolu voxel sayısını sayar.
    ConvexHull'a göre gürültüye daha dayanıklı ve concave şekillerde daha az şişer.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    if points.shape[0] < 10:
        return 0.0

    vs = float(voxel_size)
    if vs <= 0:
        raise ValueError("voxel_size must be > 0")

    # Voxel indeksleri
    q = np.floor(points / vs).astype(np.int32)
    # benzersiz voxel’ler
    uniq = np.unique(q, axis=0)
    return float(uniq.shape[0] * (vs ** 3))


def estimate_volume_m3_voxel(
    depth_ref: np.ndarray,
    depth_calib: np.ndarray,
    mask_calib: np.ndarray,
    calib_volume_m3: float,
    depth_test: np.ndarray,
    mask_test: np.ndarray,
    fov_x_deg: float = 60.0,
    fov_y_deg: float | None = None,
    voxel_size: float = 0.01,
) -> tuple[float, float, float, float]:
    """
    Returns:
      volume_test_m3, scale, rel_calib_3d, rel_test_3d
    """
    pts_c = depthdiff_to_pointcloud(depth_ref, depth_calib, mask_calib, fov_x_deg=fov_x_deg, fov_y_deg=fov_y_deg)
    pts_t = depthdiff_to_pointcloud(depth_ref, depth_test,  mask_test,  fov_x_deg=fov_x_deg, fov_y_deg=fov_y_deg)

    rel_c = relative_volume_voxel(pts_c, voxel_size=voxel_size)
    rel_t = relative_volume_voxel(pts_t, voxel_size=voxel_size)

    if rel_c <= 1e-12:
        raise RuntimeError("Calibration 3D relative volume too small")

    scale = calib_volume_m3 / rel_c
    vol_test = rel_t * scale
    return float(vol_test), float(scale), float(rel_c), float(rel_t)
