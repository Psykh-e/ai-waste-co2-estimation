# src/reporting.py
import numpy as np
import cv2

def draw_box(img_bgr, box_xyxy, label=None):
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    out = img_bgr.copy()
    cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
    if label:
        cv2.putText(out, label, (x1, max(25, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return out

def overlay_mask(img_bgr, mask_u8, color=(0,0,255), alpha=0.5):
    out = img_bgr.copy()
    m = mask_u8 > 0
    out[m] = (alpha * out[m] + (1-alpha) * np.array(color)).astype(np.uint8)
    return out

def depth_viz(depth):
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return (d * 255).astype(np.uint8)
