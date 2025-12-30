# # src/clip_classifier.py
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Dict, List, Optional

# import numpy as np
# import cv2
# import torch
# import clip
# from PIL import Image


# @dataclass
# class ClipResult:
#     label: str
#     scores: Dict[str, float]


# def _mask_to_crop(
#     img_bgr: np.ndarray,
#     mask_u8: np.ndarray,
#     pad: int = 20,
#     bg_gray: int = 127
# ) -> np.ndarray:
#     if mask_u8.dtype != np.bool_:
#         m = mask_u8 > 0
#     else:
#         m = mask_u8

#     ys, xs = np.where(m)
#     if xs.size < 10:
#         raise RuntimeError("CLIP crop: mask too small")

#     x1, x2 = xs.min(), xs.max()
#     y1, y2 = ys.min(), ys.max()

#     H, W = mask_u8.shape[:2]
#     x1 = max(0, x1 - pad)
#     y1 = max(0, y1 - pad)
#     x2 = min(W - 1, x2 + pad)
#     y2 = min(H - 1, y2 + pad)

#     crop = img_bgr[y1:y2 + 1, x1:x2 + 1].copy()
#     crop_m = m[y1:y2 + 1, x1:x2 + 1]

#     bg = np.full_like(crop, bg_gray, dtype=np.uint8)
#     crop = np.where(crop_m[..., None], crop, bg)

#     return crop


# class ClipMaterialClassifier:

#     def __init__(
#         self,
#         labels: List[str],
#         model_name: str = "ViT-B/32",
#         device: Optional[str] = None
#     ):
#         if device is None:
#             device = "cuda" if torch.cuda.is_available() else "cpu"

#         self.device = device
#         self.labels = labels

#         print("[CLIP] Loading OpenAI CLIP:", model_name)
#         self.model, self.preprocess = clip.load(model_name, device=self.device)
#         self.model.eval()

#         with torch.no_grad():
#             text_tokens = clip.tokenize(self.labels).to(self.device)
#             text_feats = self.model.encode_text(text_tokens)
#             self.text_features = text_feats / text_feats.norm(dim=-1, keepdim=True)

#     @torch.no_grad()
#     def classify(
#         self,
#         img_bgr: np.ndarray,
#         mask_u8: np.ndarray,
#         pad: int = 20,
#         bg_gray: int = 127,
#         save_crop_path: Optional[str] = None
#     ) -> ClipResult:

#         crop_bgr = _mask_to_crop(img_bgr, mask_u8, pad=pad, bg_gray=bg_gray)

#         if save_crop_path is not None:
#             cv2.imwrite(save_crop_path, crop_bgr)

#         crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
#         pil_img = Image.fromarray(crop_rgb)

#         image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)

#         image_feats = self.model.encode_image(image_input)
#         image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

#         sims = (image_feats @ self.text_features.T).squeeze(0)
#         sims_np = sims.float().cpu().numpy()

#         scores = {lab: float(sc) for lab, sc in zip(self.labels, sims_np)}
#         best_label = self.labels[int(np.argmax(sims_np))]

#         return ClipResult(label=best_label, scores=scores)



























# src/clip_classifier.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import cv2
import torch
import clip
from PIL import Image


# =================================================
# ÇIKTI
# =================================================
@dataclass
class ClipResult:
    label: str
    scores: Dict[str, float]   # sınıf -> skor


# =================================================
# PROMPT ENSEMBLE (EN KRİTİK KISIM)
# =================================================

PROMPT_ENSEMBLE = {
    "plastic object": [
    "a plastic waste object",
    "a discarded plastic item",
    "a plastic bottle waste",
    "a plastic packaging material",
    "a lightweight plastic container",
    "a single-use plastic object",
    "a plastic household waste",
    "a plastic food packaging",
    "a plastic consumer product waste"
    ],
    "metal object": [
        "a metal waste object",
        "a discarded metal item",
        "an aluminum can waste",
        "a metallic waste material",
        "a shiny metal object",
        "a steel or aluminum object",
        "a metal packaging waste",
        "a metal household waste",
        "a rigid metal object"
    ],
    "glass object": [
        "a glass waste object",
        "a discarded glass bottle",
        "a transparent glass container",
        "a clear glass packaging",
        "a fragile glass object",
        "a glass food container",
        "a glass household waste",
        "a transparent rigid object made of glass"
    ],
    "cardboard object": [
        "a cardboard waste object",
        "a discarded cardboard box",
        "a paper packaging waste",
        "a carton box made of paper",
        "a paper-based packaging material",
        "a folded cardboard container",
        "a paperboard food package",
        "a recyclable paper box"
    ]



}


# =================================================
# MASK → TEMİZ CROP (SQUARE + CENTERED)
# =================================================
def mask_to_square_crop(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    pad_ratio: float = 0.15,
    bg_gray: int = 127
) -> np.ndarray:

    mask = mask_u8 > 0
    ys, xs = np.where(mask)
    if len(xs) < 20:
        raise RuntimeError("CLIP crop: mask too small")

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    h = y2 - y1 + 1
    w = x2 - x1 + 1
    side = int(max(h, w) * (1 + pad_ratio))

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    H, W = img_bgr.shape[:2]
    half = side // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(W - 1, cx + half)
    y2 = min(H - 1, cy + half)

    crop = img_bgr[y1:y2 + 1, x1:x2 + 1].copy()
    crop_mask = mask[y1:y2 + 1, x1:x2 + 1]

    bg = np.full_like(crop, bg_gray, dtype=np.uint8)
    crop = np.where(crop_mask[..., None], crop, bg)

    return crop


# =================================================
# CLIP SINIFLANDIRICI
# =================================================
class ClipMaterialClassifier:

    def __init__(
        self,
        labels: List[str],
        model_name: str = "ViT-B/16",   # 🔥 B/32 yerine B/16
        device: Optional[str] = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.labels = labels

        print("[CLIP] Loading OpenAI CLIP:", model_name)
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # -------------------------
        # TEXT EMBEDDING (PROMPT ENSEMBLE)
        # -------------------------
        self.text_features = {}

        with torch.no_grad():
            for label in self.labels:
                prompts = PROMPT_ENSEMBLE[label]
                tokens = clip.tokenize(prompts).to(self.device)

                feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)

                # prompt ortalaması
                self.text_features[label] = feats.mean(dim=0, keepdim=True)

    @torch.no_grad()
    def classify(
        self,
        img_bgr: np.ndarray,
        mask_u8: np.ndarray,
        save_crop_path: Optional[str] = None
    ) -> ClipResult:

        # -------------------------
        # CROP
        # -------------------------
        crop_bgr = mask_to_square_crop(img_bgr, mask_u8)

        if save_crop_path:
            cv2.imwrite(save_crop_path, crop_bgr)

        # -------------------------
        # KONTRAST İYİLEŞTİRME (HAFİF)
        # -------------------------
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        crop_bgr = cv2.merge([l, a, b])
        crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_LAB2BGR)

        # -------------------------
        # CLIP IMAGE EMBEDDING
        # -------------------------
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        image_feat = self.model.encode_image(image_input)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        # -------------------------
        # COSINE SIM + ENSEMBLE
        # -------------------------
        scores = {}
        for label, txt_feat in self.text_features.items():
            sim = float((image_feat @ txt_feat.T).item())
            scores[label] = sim

        best_label = max(scores, key=scores.get)

        return ClipResult(
            label=best_label,
            scores=scores
        )
