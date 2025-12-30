<p align="center">
  <h1 align="center">AI-Based Waste Volume, Mass & CO₂ Estimation</h1>
  <p align="center">
    Monocular Computer Vision Pipeline for Sustainable Waste Analysis
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red">
  <img src="https://img.shields.io/badge/Computer%20Vision-MiDaS%20%7C%20YOLOv8%20%7C%20CLIP-green">
  <img src="https://img.shields.io/badge/Status-Graduation%20Project-success">
</p>

---

## 📌 Overview

This project presents an **end-to-end AI-based computer vision system** that estimates the  
**volume**, **mass**, and **CO₂-equivalent emission savings** of waste materials using a  
**single RGB image**.

The system is developed as a **Mechatronics Engineering Graduation Project** and focuses on  
**sustainability-oriented waste analysis without requiring additional sensors** such as  
stereo cameras or LiDAR.

All geometric information is inferred using **monocular depth estimation**.

---

## 🎯 Objectives

The main objectives of this study are:

- Detect waste objects from RGB images  
- Perform pixel-level object segmentation  
- Classify waste materials by type  
- Estimate object volume using depth information  
- Compute object mass using material density  
- Calculate potential CO₂-equivalent emission savings  

---

## 🧠 System Architecture & Methods

The proposed system is composed of modular stages, each responsible for a specific task.

| Stage | Method | Purpose |
|------|------|------|
| Depth Estimation | MiDaS | Extract relative depth from a single RGB image |
| Object Segmentation | YOLOv8-Seg | Detect and segment waste objects |
| Material Classification | CLIP | Classify object material using vision-language similarity |
| Volume Estimation | Depth Integration / Voxel Method | Estimate object volume |
| Mass Estimation | Density × Volume | Compute physical mass |
| CO₂ Estimation | Literature-based factors | Estimate recycling emission savings |

---

## ♻️ Supported Waste Materials

The system currently supports the following waste categories:

| Material | Description |
|--------|------------|
| Plastic | Common household plastic waste |
| Metal | Aluminum and steel (representative materials) |
| Glass | Soda-lime / float glass |
| Cardboard / Paper | Paper-based packaging waste |

Density values are selected from **peer-reviewed scientific literature** and  
commonly accepted industrial reference data.

---

## 📁 Project Structure

    Project/
    ├── data/
    │   ├── calib.jpeg        # Calibration image
    │   ├── ref.jpeg          # Reference depth image
    │   └── tests/            # Test images
    │
    ├── src/
    │   ├── depth_midas.py        # MiDaS depth estimation
    │   ├── segment_yolo.py       # YOLOv8 segmentation
    │   ├── clip_classifier.py    # CLIP-based classification
    │   ├── volume_integral.py    # Volume estimation (depth integration)
    │   ├── volume_pointcloud.py # Voxel-based volume estimation
    │   ├── mass_estimator.py     # Mass & CO₂ calculation
    │   ├── reporting.py          # Visualization utilities
    │   └── io_utils.py           # Input / output utilities
    │
    ├── main.py                   # Main pipeline execution
    ├── requirements.txt
    ├── .gitignore
    └── README.md

---

## ⚙️ Installation

Create a Python environment and install the required dependencies:

    conda create -n waste-ai python=3.10
    conda activate waste-ai
    pip install -r requirements.txt

A CUDA-enabled GPU is strongly recommended for improved performance, especially for  
depth estimation and segmentation tasks.

---

## ▶️ Running the Project

To execute the full pipeline, run:

    python main.py

The pipeline automatically performs the following steps:

- Generates monocular depth maps  
- Segments waste objects  
- Classifies material types  
- Estimates object volume  
- Computes object mass  
- Calculates CO₂-equivalent emission savings  
- Saves outputs for visualization and analysis  

---

## 📐 Volume & Mass Estimation

Mass estimation is performed using the standard physical relation:

    Mass (kg) = Volume (m³) × Density (kg/m³)

Two alternative volume estimation approaches are implemented:

**Depth Integration Method**  
The object volume is estimated by integrating depth values over the segmented object region.

**Voxel-Based Point Cloud Method**  
Depth information is converted into a 3D point cloud, and volume is estimated by voxel occupancy.

---

## 🌍 CO₂ Emission Saving Estimation

CO₂-equivalent emission savings are calculated as:

    CO₂ Saving (kg CO₂e) = Mass (kg) × CO₂e Factor (kg CO₂e/kg)

CO₂ emission factors are obtained from **peer-reviewed environmental and recycling literature**.

---

## 📦 Model Weights

YOLOv8 segmentation model weight files are not included in this repository due to  
GitHub file size limitations.

Model files can be downloaded automatically by Ultralytics or manually placed under:

    models/
      yolov8l-seg.pt
      yolov8m-seg.pt
      yolov8x-seg.pt

---

## ⚠️ Limitations

- Monocular depth estimation provides **relative**, not absolute, depth information  
- Density values are **representative averages**  
- Accuracy depends on calibration quality and scene conditions  
- Reflective or transparent materials may affect depth accuracy  

---

## 🎓 Academic Context

This project integrates **Artificial Intelligence**, **Computer Vision**, and  
**Environmental Sustainability** to contribute to intelligent waste analysis systems.

The study is conducted as a **Mechatronics Engineering Graduation Project**.

---

## 👤 Author

**Berkay Yılmaz**  
Mechatronics Engineering  
Graduation Project – 2025  

---

## 📜 License

This project is intended for **academic and research purposes only**.  
Commercial use requires explicit permission from the author.
