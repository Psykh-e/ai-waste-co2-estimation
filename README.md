# AI-Based Waste Material Detection, Volume, Mass and CO₂ Saving Estimation

This project presents an **end-to-end AI-based computer vision system** that estimates the **volume, mass, and CO₂-equivalent emission savings** of waste materials using a **single RGB image**.

The system is developed as a **Mechatronics Engineering Graduation Project** and focuses on sustainability-oriented waste analysis without requiring additional sensors such as stereo cameras or LiDAR.

---

## Project Overview

The pipeline automatically performs:

1. Monocular depth estimation  
2. Waste object segmentation  
3. Material classification  
4. Volume estimation  
5. Mass calculation  
6. CO₂ emission saving estimation  

All steps are integrated into a fully automated pipeline.

---

## Objectives

- Detect waste objects from RGB images  
- Classify materials (plastic, metal, glass, cardboard)  
- Estimate object volume using depth information  
- Compute object mass using material density  
- Calculate potential CO₂-equivalent savings through recycling  

---

## Methods and Models

| Stage | Method / Model |
|-----|---------------|
| Depth Estimation | MiDaS |
| Object Segmentation | YOLOv8 (Segmentation) |
| Material Classification | CLIP |
| Volume Estimation | Depth Integration / Voxel-based Point Cloud |
| Mass Estimation | Density × Volume |
| CO₂ Saving Estimation | Literature-based CO₂e factors |

---

## Supported Materials

- Plastic  
- Metal (aluminum / steel representative)  
- Glass  
- Cardboard / Paper  

Material density values are selected based on scientific literature and common industrial standards.

---

## Project Structure

```
.
├── data/                     # Input images (ignored in git)
├── runs/                     # Output results (ignored in git)
│
├── src/
│   ├── depth_midas.py
│   ├── segment_yolo.py
│   ├── clip_classifier.py
│   ├── volume_integral.py
│   ├── volume_pointcloud.py
│   ├── mass_estimator.py
│   ├── reporting.py
│   └── io_utils.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```
conda create -n waste-ai python=3.10
conda activate waste-ai
pip install -r requirements.txt
```

---

## Running the Project

```
python main.py
```

The pipeline automatically generates depth maps, segmentation masks, volume and mass estimates, and CO₂ saving results.

---

## Volume and Mass Calculation

```
Mass (kg) = Volume (m³) × Density (kg/m³)
```

---

## CO₂ Saving Calculation

```
CO₂ Saving (kg CO₂e) = Mass (kg) × CO₂e Factor (kg CO₂e/kg)
```

---

## Model Weights

YOLOv8 model weight files (.pt) are not included in this repository due to GitHub file size limitations.

Models can be automatically downloaded by Ultralytics or manually placed under:

```
models/
 ├── yolov8l-seg.pt
 ├── yolov8m-seg.pt
 └── yolov8x-seg.pt
```

---

## Limitations

- Monocular depth estimation provides relative depth
- Density values are representative averages
- Accuracy depends on calibration and scene conditions

---

## Future Work

- Multi-object support  
- Stereo / LiDAR integration  
- Real-time video processing  

---

## Author

Berkay Yılmaz  
Mechatronics Engineering  
Graduation Project – 2025

---

## License

This project is intended for academic and research purposes only.
