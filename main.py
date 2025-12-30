import os
import json
from datetime import datetime
import cv2
import numpy as np

from src.io_utils import load_image, ensure_dir, save_depth_npy
from src.depth_midas import MiDaSDepthEstimator
from src.reporting import overlay_mask
from src.segment_yolo import YoloDepthSegmenter
from src.clip_classifier import ClipMaterialClassifier

from src.volume_integral import estimate_volume_m3
from src.volume_pointcloud import estimate_volume_m3_voxel
from src.mass_estimator import estimate_mass_scenarios
from src.co2_estimator import estimate_co2_scenarios 


# -------------------------------------------------
# AYARLAR
# -------------------------------------------------
DATA_DIR = "data"
RUNS_DIR = "runs"

CALIB_VOLUME_M3 = 0.000216
CAMERA_FOV_DEG  = 60.0
VOXEL_SIZE      = 0.005   # 5 mm


# -------------------------------------------------
# CLIP LABELS
# -------------------------------------------------
CLIP_LABELS = [
    "plastic object",
    "metal object",
    "glass object",
    "cardboard object"
]


# -------------------------------------------------
# DEPTH GÖRSELLEŞTİRME (MAGMA / INFERNO)
# -------------------------------------------------
def save_depth_colormap(depth, out_path, cmap="magma"):
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    d_u8 = (d * 255).astype(np.uint8)

    if cmap == "inferno":
        colored = cv2.applyColorMap(d_u8, cv2.COLORMAP_INFERNO)
    else:
        colored = cv2.applyColorMap(d_u8, cv2.COLORMAP_MAGMA)

    cv2.imwrite(out_path, colored)


# -------------------------------------------------
# RUN DİZİNİ
# -------------------------------------------------
def create_run_dir():
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RUNS_DIR, run_id)

    for sub in ["depth", "masks", "metrics"]:
        ensure_dir(os.path.join(run_dir, sub))

    return run_dir


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():

    run_dir = create_run_dir()
    depth_dir = os.path.join(run_dir, "depth")
    mask_dir  = os.path.join(run_dir, "masks")
    metrics_path = os.path.join(run_dir, "metrics", "results.json")

    # -------------------------
    # MODELLER
    # -------------------------
    midas = MiDaSDepthEstimator()
    segmenter = YoloDepthSegmenter(
        model_path="models/yolov8x-seg.pt",
        conf=0.15
    )
    clipper = ClipMaterialClassifier(labels=CLIP_LABELS)

    # -------------------------
    # REF
    # -------------------------
    ref_img = load_image(os.path.join(DATA_DIR, "ref.jpeg"))
    ref_depth = midas.compute_depth(ref_img)
    save_depth_npy(ref_depth, os.path.join(depth_dir, "ref.npy"))
    save_depth_colormap(ref_depth, os.path.join(depth_dir, "ref.png"))

    # -------------------------
    # CALIB
    # -------------------------
    calib_img = load_image(os.path.join(DATA_DIR, "calib.jpeg"))
    calib_depth = midas.compute_depth(calib_img)
    save_depth_npy(calib_depth, os.path.join(depth_dir, "calib.npy"))
    save_depth_colormap(calib_depth, os.path.join(depth_dir, "calib.png"))

    calib_mask = segmenter.segment(
        img_bgr=calib_img,
        depth_ref=ref_depth,
        depth_tgt=calib_depth
    )
    cv2.imwrite(os.path.join(mask_dir, "calib.png"), calib_mask * 255)

    # -------------------------
    # TESTLER
    # -------------------------
    results = {}
    tests_dir = os.path.join(DATA_DIR, "tests")

    for fname in sorted(os.listdir(tests_dir)):
        img_path = os.path.join(tests_dir, fname)
        name = os.path.splitext(fname)[0]

        print(f"\n[PROCESS] {name}")

        img = load_image(img_path)
        depth = midas.compute_depth(img)

        save_depth_npy(depth, os.path.join(depth_dir, f"{name}.npy"))
        save_depth_colormap(depth, os.path.join(depth_dir, f"{name}.png"))

        # -------------------------
        # MASKE
        # -------------------------
        mask = segmenter.segment(
            img_bgr=img,
            depth_ref=ref_depth,
            depth_tgt=depth
        )

        cv2.imwrite(os.path.join(mask_dir, f"{name}.png"), mask * 255)
        cv2.imwrite(
            os.path.join(mask_dir, f"{name}_overlay.png"),
            overlay_mask(img, mask)
        )

        # -------------------------
        # HACİM
        # -------------------------
        vol_int, _, _, _ = estimate_volume_m3(
            depth_ref=ref_depth,
            depth_calib=calib_depth,
            mask_calib=calib_mask > 0,
            calib_volume_m3=CALIB_VOLUME_M3,
            depth_test=depth,
            mask_test=mask > 0
        )

        vol_vox, _, _, _ = estimate_volume_m3_voxel(
            depth_ref=ref_depth,
            depth_calib=calib_depth,
            mask_calib=calib_mask > 0,
            calib_volume_m3=CALIB_VOLUME_M3,
            depth_test=depth,
            mask_test=mask > 0,
            fov_x_deg=CAMERA_FOV_DEG,
            voxel_size=VOXEL_SIZE
        )

        # -------------------------
        # CLIP SINIFLANDIRMA
        # -------------------------
        clip_res = clipper.classify(img, mask > 0)
        material = clip_res.label
        clip_scores = clip_res.scores

        print(f"  Material: {material}")
        print("  CLIP scores:")
        for k, v in clip_scores.items():
            print(f"    {k:18s}: {v:.3f}")

        # -------------------------
        # KÜTLE (3 SENARYO)
        # -------------------------
        mass_res = estimate_mass_scenarios(
            volume_m3=vol_vox,
            material_label=material
        )

        print(
            f"  Mass (kg) → "
            f"low: {mass_res['mass_low_kg']:.3f}, "
            f"mid: {mass_res['mass_mid_kg']:.3f}, "
            f"high: {mass_res['mass_high_kg']:.3f}"
        )

        # -------------------------
        # CO₂e TASARRUF (3 SENARYO)
        # -------------------------
        co2_res = estimate_co2_scenarios(
            material_label=material,
            mass_low_kg=mass_res["mass_low_kg"],
            mass_mid_kg=mass_res["mass_mid_kg"],
            mass_high_kg=mass_res["mass_high_kg"]
        )

        print(
            f"  CO2e (kg) → "
            f"low: {co2_res['co2_low_kg']:.3f}, "
            f"mid: {co2_res['co2_mid_kg']:.3f}, "
            f"high: {co2_res['co2_high_kg']:.3f}"
        )

        # -------------------------
        # JSON KAYIT
        # -------------------------
        results[name] = {
            "volume_integral_m3": float(vol_int),
            "volume_voxel_m3": float(vol_vox),

            "material": material,
            "clip_scores": clip_scores,

            **mass_res,
            **co2_res
        }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Run completed → {run_dir}")
    print(f"[OK] Results saved → results.json")


if __name__ == "__main__":
    main()
