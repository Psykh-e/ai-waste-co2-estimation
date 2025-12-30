# src/segment_yolo.py

import cv2
import numpy as np
from ultralytics import YOLO


YOLO_MODEL_PATH = "../models/yolov8x-seg.pt"
YOLO_CONF = 0.1

MIN_AREA = 500
MAX_AREA_RATIO = 0.40

BANNED_CLASS_NAMES = {
    "dining table", "chair", "couch", "sofa", "bed", "bench",
    "tv", "tvmonitor", "refrigerator"
}


def robust_norm_to_u8(x, p_lo=2, p_hi=98):
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    return (x * 255).astype(np.uint8)


def depth_fallback_mask(depth_ref, depth_tgt):
    """
    YOLO maske yoksa:
    Ref–Target depth farkından nesne maskesi çıkar
    """
    H, W = depth_tgt.shape

    diff = np.abs(depth_ref - depth_tgt)
    diff_u8 = robust_norm_to_u8(diff)

    blur = cv2.GaussianBlur(diff_u8, (7, 7), 0)
    _, mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    best = None
    best_area = 0

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best = i

    out = np.zeros((H, W), np.uint8)
    if best is not None:
        out[labels == best] = 1

    return out


class YoloDepthSegmenter:

    def __init__(self, model_path=YOLO_MODEL_PATH, conf=YOLO_CONF):
        print("[YOLO-SEG] Loading model:", model_path)
        self.model = YOLO(model_path)
        self.conf = conf

    def segment(self, img_bgr, depth_ref, depth_tgt):
        H, W = img_bgr.shape[:2]

        # ---------- DEPTH ADAY ----------
        diff = np.abs(depth_ref - depth_tgt)
        diff_u8 = robust_norm_to_u8(diff)
        diff_blur = cv2.GaussianBlur(diff_u8, (7, 7), 0)
        _, cand = cv2.threshold(
            diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        cand_bool = cand > 0

        # ---------- YOLO ----------
        results = self.model.predict(img_bgr, conf=self.conf, verbose=False)
        res = results[0]

        # 🚑 YOLO HİÇ MASKE VERMEDİ → FALLBACK
        if res.masks is None or res.masks.data is None:
            print("[YOLO-SEG] YOLO maske yok → depth fallback")
            return depth_fallback_mask(depth_ref, depth_tgt)

        masks = res.masks.data.cpu().numpy()
        boxes = res.boxes
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        best_mask = None
        best_score = -1.0

        for i in range(masks.shape[0]):
            cls_id = int(cls_ids[i])
            class_name = res.names[cls_id]

            if class_name in BANNED_CLASS_NAMES:
                continue

            m = cv2.resize(masks[i], (W, H), interpolation=cv2.INTER_LINEAR)
            m_bin = m > 0.5

            num, labels, stats, cents = cv2.connectedComponentsWithStats(
                m_bin.astype(np.uint8)
            )

            for lbl in range(1, num):
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area < MIN_AREA:
                    continue
                if area > MAX_AREA_RATIO * H * W:
                    continue

                blob = labels == lbl
                overlap = np.logical_and(blob, cand_bool).sum() / (area + 1e-6)

                score = confs[i] * overlap * area

                if score > best_score:
                    best_score = score
                    best_mask = np.zeros((H, W), np.uint8)
                    best_mask[blob] = 1

        # 🚑 YOLO VAR AMA SEÇİLEMEDİ → FALLBACK
        if best_mask is None:
            print("[YOLO-SEG] Uygun YOLO maskesi yok → depth fallback")
            return depth_fallback_mask(depth_ref, depth_tgt)

        kernel = np.ones((5, 5), np.uint8)
        best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return best_mask