# src/depth_midas.py

from pathlib import Path
import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F


class MiDaSDepthEstimator:
    """
    MiDaS tabanlı derinlik çıkarıcı.
    - GPU + FP16 destekli
    - Downscale -> inference -> upscale
    """

    def __init__(
        self,
        model_type: str = "DPT_Large",
        max_side: int = 1024,
        use_fp16: bool = True,
        local_repo: str | None = None,
        device: str | None = None,
    ):
        self.model_type = model_type
        self.max_side = max_side
        self.use_fp16 = use_fp16

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        torch.backends.cudnn.benchmark = True
        torch.set_grad_enabled(False)

        self._load_model(local_repo)

    # --------------------------
    # Model yükleme
    # --------------------------
    def _load_model(self, local_repo: str | None):
        print(f"[MiDaS] Loading model: {self.model_type}")

        if local_repo and os.path.isdir(local_repo):
            midas = torch.hub.load(local_repo, self.model_type, source="local")
            transforms = torch.hub.load(local_repo, "transforms", source="local")
        else:
            midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        self.model = midas.to(self.device).eval()

        if self.device == "cuda" and self.use_fp16:
            self.model = self.model.half()

        if self.model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

        print(f"[MiDaS] Device: {self.device}, FP16: {self.use_fp16}")

    # --------------------------
    # Yardımcılar
    # --------------------------
    @staticmethod
    def _resize_max_side(img_bgr: np.ndarray, max_side: int):
        h, w = img_bgr.shape[:2]
        if max(h, w) <= max_side:
            return img_bgr, 1.0

        scale = max_side / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_small = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img_small, scale

    def _autocast(self):
        if self.device == "cuda" and self.use_fp16:
            return torch.amp.autocast("cuda")
        from contextlib import nullcontext
        return nullcontext()

    # --------------------------
    # Ana fonksiyon
    # --------------------------
    def compute_depth(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Girdi: BGR görüntü (H,W,3)
        Çıktı: depth (H,W) float32
        """
        h_full, w_full = img_bgr.shape[:2]

        # 1) Downscale
        img_small, _ = self._resize_max_side(img_bgr, self.max_side)

        # 2) Preprocess
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        inp = self.transform(img_rgb).to(self.device)

        if self.device == "cuda" and self.use_fp16:
            inp = inp.half()

        # 3) Inference
        with self._autocast():
            pred = self.model(inp)
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=img_small.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        depth_small = pred.float().cpu().numpy()

        # 4) Upscale -> orijinal boyut
        depth_full = cv2.resize(
            depth_small,
            (w_full, h_full),
            interpolation=cv2.INTER_CUBIC
        )

        return depth_full.astype(np.float32)

    # --------------------------
    # Kayıt yardımcıları
    # --------------------------
    @staticmethod
    def save_depth_npy(path: Path, depth: np.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), depth)

    @staticmethod
    def save_depth_png(path: Path, depth: np.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        d = depth.astype(np.float32)
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        d = (d * 255).astype(np.uint8)
        cv2.imwrite(str(path), d)
