# Waste Classification with EfficientNetB0 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

This project implements an end-to-end deep learning pipeline for **waste image classification** using **EfficientNetB0**, trained and evaluated entirely in a **Kaggle Notebook**. The workflow includes automatic data cleaning, stratified train/val/test splitting, robust augmentation, mixed-precision training, and detailed performance analysis — all optimized for the Kaggle environment with GPU acceleration.

---

## Overview

Given a dataset of waste images (e.g., cardboard, glass, metal, paper, plastic, trash), this notebook:
- Filters out corrupted or too-small images
- Splits data into **70% train / 15% validation / 15% test**
- Trains an **ImageNet-pretrained EfficientNetB0**
- Uses **AdamW + Cosine Annealing** for stable convergence
- Evaluates performance with **accuracy, confusion matrix, and classification report**
- Visualizes **training curves** and **misclassified samples**

The entire pipeline runs in a single Kaggle notebook with **no manual intervention**.

---

## Dataset

- **Source**: [Waste Classification – Kaggle Dataset](https://www.kaggle.com/datasets/adithyachalla/waste-classification)  
- **Classes**: Automatically detected from subfolder names  
  Example: `['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']`
- **Total Images**: ~2,500 (varies slightly after cleaning)

### Preprocessing Steps
✅ Skip files < 10 KB (likely corrupt)  
✅ Convert all images to **RGB**  
✅ Resize to **224×224**  
✅ Preserve original filenames for traceability  
✅ Reproducible split using fixed random seed (`seed=42`)

The cleaned dataset is saved to:
```
./waste_clean_dataset/
├── train/
├── val/
└── test/
```

---

## Model Architecture

- **Backbone**: `torchvision.models.efficientnet_b0`
- **Pretrained Weights**: ImageNet1K (`IMAGENET1K_V1`)
- **Modified Head**:
  ```python
  nn.Sequential(
      nn.Dropout(p=0.4),
      nn.Linear(in_features=1280, out_features=num_classes)
  )

* Loss Function: CrossEntropyLoss
* Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
* Learning Rate Scheduler: CosineAnnealingLR (T_max = total epochs)
* Precision: Mixed-precision training via torch.cuda.amp

## Data Augmentation

- RandomResizedCrop(224, scale=(0.7, 1.0))
- RandomHorizontalFlip(p=0.5)
- RandomVerticalFlip(p=0.1)
- RandomRotation(±10°)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)
- Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Validation & Test Sets
- Resize(256)
- CenterCrop(224)
- ToTensor()
- Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Training Configuration

| Parameter | Nilai Default | Keterangan |
| :--- | :--- | :--- |
| **Epochs** | 15 | Jumlah iterasi penuh pelatihan melalui seluruh dataset. |
| **Batch Size** | 32 | Jumlah sampel gambar yang diproses per iterasi selama pelatihan. |
| **Image Size** | 224×224 | Resolusi akhir gambar input setelah augmentasi dan pra-pemrosesan. |
| **Optimizer** | AdamW | Optimizer Adam dengan weight decay terpisah, cocok untuk fine-tuning model pretrained. |
| **Initial Learning Rate** | 0.001 | Laju pembelajaran awal sebelum dijadwalkan menurun. |
| **Weight Decay** | 1e-4 | Regularisasi L2 untuk mencegah overfitting pada bobot model. |
| **LR Scheduler** | Cosine Annealing | Menurunkan learning rate mengikuti bentuk kurva kosinus selama pelatihan. |
| **Loss Function** | CrossEntropyLoss | Fungsi kerugian standar untuk klasifikasi multi-kelas. |
| **Random Seed** | 42 | Nilai seed untuk memastikan hasil yang dapat direproduksi (PyTorch, NumPy, Python). |
| **GPU** | P100 / T4 (Kaggle) | Akselerasi pelatihan menggunakan GPU yang tersedia di lingkungan Kaggle. |

- Best model is saved based on **validation accuracy**  
- Checkpoints stored in `./effnet_waste_output/`  
- Full training history (loss & accuracy) logged for analysis

## Evaluation Metrics & Visualizations

After training, the notebook automatically generates:

- ✅ **Final Test Accuracy**  
- ✅ **Per-class Precision, Recall, F1-Score** (via `classification_report`)  
- ✅ **Confusion Matrix Heatmap**  
- ✅ **Training vs Validation Curves** (loss & accuracy over epochs)  
- ✅ **Sample Training Images** (with true labels)  
- ✅ **Top 10 Misclassified Test Images** (for qualitative error analysis)

All plots are rendered inline using `matplotlib`.

---

## How to Run on Kaggle

1. **Open a new Kaggle Notebook**  
2. **Add the dataset**:  
   - Click **“+ Add data”** in the notebook editor  
   - Search for **“Waste Classification”** by *Adithya Challa*  
   - Click **Add** to attach it to your notebook  
3. **Enable GPU**:  
   - Go to **Notebook Settings → Accelerator → GPU (P100 or T4)**  
4. **Paste the entire script** into a single code cell  
5. **Run all cells**

> 💡 **No additional libraries need to be installed** — everything uses Kaggle’s default environment (PyTorch, torchvision, scikit-learn, matplotlib, etc.).
