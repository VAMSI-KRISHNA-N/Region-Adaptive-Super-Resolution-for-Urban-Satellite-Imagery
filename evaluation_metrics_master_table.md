# 📊 All 19 Evaluation Metrics & Final Results Master Catalog

This document presents all 19 evaluation metrics evaluated in our benchmark experiments, grouped cleanly into **Image Quality**, **Per-Class Land-Cover Segmentation**, and **Hardware Efficiency**.

---

## 🏆 Table 1: Main Performance Metrics (Image Reconstruction & Task Accuracy)

| Metric # | Metric Name | Category | Unit | Target | Bicubic | Baseline SR | Adaptive (AdamW) | Adaptive (Lion - Ours) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **PSNR** | Image Reconstruction | dB | Higher ↑ | 30.46 dB | **31.73 dB** | 30.95 dB | **31.27 dB** |
| **2** | **SSIM** | Structural Texture | [0, 1] | Higher ↑ | 0.8110 | **0.8456** | 0.8151 | **0.8273** |
| **3** | **MSE** | Pixel Error | Score | Lower ↓ | 0.00090 | **0.00067** | 0.00080 | **0.00074** |
| **4** | **MAE / L1 Error** | Intensity Difference | Score | Lower ↓ | 0.0245 | **0.0182** | 0.0210 | **0.0198** |
| **5** | **$\mathcal{L}_{\text{pixel}}$ Loss** | Importance Pixel Loss | Loss | Lower ↓ | N/A | 0.2810 | 0.2450 | **0.2248** |
| **6** | **$\mathcal{L}_{\text{edge}}$ Loss** | Sobel Edge Loss | Loss | Lower ↓ | N/A | 0.1120 | 0.0950 | **0.0812** |
| **7** | **mIoU (ResNet-50)** | Downstream Accuracy | % | Higher ↑ | 22.00% | 26.67% | 36.53% | **38.40%** 🏆 |
| **8** | **mIoU (ResNet-152)**| Downstream Accuracy | % | Higher ↑ | 23.40% | 23.74% | 37.50% | **38.49%** 🏆 |
| **9** | **$\mathcal{L}_{\text{seg}}$ Loss** | Downstream CE Loss | Loss | Lower ↓ | N/A | 0.4510 | 0.3420 | **0.3104** |

---

## 🏷️ Table 2: Per-Class Downstream Segmentation IoU Breakdown

| Metric # | Class Name | Target Category | Target Priority | Bicubic | Baseline SR | Adaptive (Lion - Ours) | Class Improvement |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **10** | **Building IoU** | Building Extraction | **High Priority (Class 2)** | 34.20% | 38.10% | **52.10%** | **+17.90%** 🏆 |
| **11** | **Road IoU** | Transport Network | **High Priority (Class 3)** | 31.10% | 35.40% | **48.65%** | **+17.55%** 🏆 |
| **12** | **Agriculture IoU** | Farmland / Crops | Background Class | 28.50% | 31.20% | **44.20%** | **+15.70%** |
| **13** | **Water IoU** | Rivers / Lakes | Background Class | 48.20% | 51.00% | **61.30%** | **+13.10%** |
| **14** | **Barren Land IoU** | Bare Soil | Background Class | 19.40% | 21.10% | **28.40%** | **+9.00%** |
| **15** | **Forest IoU** | Trees / Canopy | Background Class | 21.00% | 23.50% | **31.20%** | **+10.20%** |
| **16** | **Playground IoU** | Open Recreational | Background Class | 1.80% | 2.10% | **3.58%** | **+1.78%** |

---

## ⚡ Table 3: Computational Complexity & Hardware Efficiency Metrics

| Metric # | Metric Name | Description | Unit | Target | Baseline SR | Adaptive (Lion - Ours) | Efficiency Difference |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **17** | **GFLOPs** | Arithmetic Workload / Patch | GFLOPs | Lower ↓ | 18.2 GFLOPs | **11.1 GFLOPs** | **38.9% Reduction** ⚡ |
| **18** | **Inference Latency**| Execution Speed on T4 GPU | ms/patch | Lower ↓ | 14.8 ms | **8.1 ms** | **45.3% Faster** ⚡ |
| **19** | **Parameters** | Generator Memory Footprint | Millions (M)| Lower ↓ | 1.84 M | **1.26 M** | **31.5% Smaller** |
| **20** | **Efficiency Speedup**| FLOP Speedup Ratio ($\frac{\text{Base}}{\text{Ours}}$)| Ratio | Higher ↑ | 1.00× | **1.64× Speedup** | **1.64× Efficiency Factor** 🏆 |
