
# 🫁 MSW-EnsNet: Multi-Scale Weighted Ensemble of CNN Architectures with Test-Time Augmentation for Robust Multi-Class Pneumonia Classification

<p align="center">
  <img src="https://github.com/user-attachments/assets/dd123697-f9f5-4c7f-8cf6-eba83fe645c2" width="234" height="167" alt="Pneumonia Research Banner"/>
</p>

<p align="center">
  <b>A systematic comparative study of Convolutional Neural Networks, Vision Transformers, and a Hybrid-ViT architecture on balanced, separately-augmented chest X-ray data for multi-class pneumonia detection.</b>
</p>

<p align="center">
  <a href="https://github.com/venkatsaikondra/Pneumonia_Research"><img src="https://img.shields.io/badge/GitHub-Code-blue?logo=github" alt="GitHub"/></a>
  <a href="https://www.kaggle.com/datasets/mohamedasak/chest-x-ray-6-classes-dataset"><img src="https://img.shields.io/badge/Dataset-Kaggle-orange?logo=kaggle" alt="Kaggle"/></a>
  <img src="https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Accuracy-95.43%25-green" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/Macro_F1-0.9544-brightgreen" alt="F1"/>
</p>

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Architecture](#architecture)
- [Results](#results)
- [Ablation Study](#ablation-study)
- [Explainability](#explainability)
- [Computational Complexity](#computational-complexity)
- [External Validation](#external-validation)
- [Installation & Usage](#installation--usage)

---

## Introduction

Pneumonia is a respiratory infection caused by either viral or bacterial pathogens. In humans, the lungs contain tiny sacs called alveoli, which fill with air during healthy breathing. When pneumonia occurs, these alveoli become filled with pus and fluid, causing discomfort, difficulty in breathing, and restricted oxygen intake.

Statistically, pneumonia impacts approximately **1,400 children per 100,000**, contributing to **14% of fatalities** among children below five years of age in 2019, resulting in the deaths of **740,180 children**. In 2017, pneumonia caused over **808,000 deaths** among children under five, representing 15% of all fatalities in this age group.

The primary diagnostic tool for pneumonia is chest X-rays, where areas of infection are identified by white patches in the pulmonary region.

### Why Multi-Class Classification Matters

Treatment pathways differ substantially across pneumonia types:
- **Bacterial pneumonia** → requires antibiotic therapy
- **Viral pneumonia** → managed symptomatically
- **COVID-19 pneumonia** → requires specialized protocols

Misclassification has direct clinical consequences: a bacterial infection classified as viral risks disease progression through antibiotic omission, while a viral infection classified as bacterial contributes to antimicrobial resistance.

### Architecture Overview

| Architecture | Description |
|---|---|
| **CNN** | Examines connections between neighboring pixels within a receptive field. Enhanced with attention mechanisms to capture both local and global dependencies. |
| **Vision Transformer (ViT)** | Abandons convolutions entirely in favor of Self-Attention mechanisms. Treats an image as a sequence of patches (visual tokens), allowing every pixel to interact with every other simultaneously. |
| **Hybrid-ViT** | Combines a CNN backbone (ResNet/BiT) to extract low-level spatial features with Transformer blocks to process those features with global attention — current state-of-the-art baseline. |

---

## Dataset

### 📌 Classes

| Class | Description |
|-------|-------------|
| `Normal` | Healthy chest X-ray |
| `Viral Pneumonia` | Pneumonia caused by viral infection |
| `Bacterial Pneumonia` | Pneumonia caused by bacterial infection |
| `COVID-19 Pneumonia` | Pneumonia associated with SARS-CoV-2 |

### Data Source

Data collected from [Kaggle — ChestX6 Multi-Class Chest X-ray Dataset](https://www.kaggle.com/datasets/mohamedasak/chest-x-ray-6-classes-dataset)

### Class Balancing

To eliminate class imbalance and avoid biased model learning, all classes were downsampled to the minimum class size (2,699):

| Class | Original Count | Balanced Count |
|-------|---------------|----------------|
| COVID-19 | 2,717 | 2,699 |
| Normal | 2,971 | 2,699 |
| Bacterial Pneumonia | 2,699 | 2,699 |
| Viral Pneumonia | 2,713 | 2,699 |
| **Total** | **11,100** | **10,796** |

### Dataset Split

| Split | Percentage | Images per Class | Total |
|-------|-----------|-----------------|-------|
| Train | 70% | 1,889 | 7,556 |
| Validation | 15% | 405 | 1,620 |
| Test | 15% | 405 | 1,620 |

### Directory Structure

```
Final_Data/
    ├── train/
    ├── val/
    └── test/
        ├── Covid-19/
        ├── Normal/
        ├── Pneumonia-Bacterial/
        └── Pneumonia-Viral/
```

---

## Methodology

### 1. Image Preprocessing

Each image undergoes the following preprocessing using OpenCV:

| Step | Operation | Details |
|------|-----------|---------|
| 1 | Load in Grayscale | Single-channel, removes color noise |
| 2 | CLAHE Enhancement | Tile: 8×8, Clip Limit: 2.0 |
| 3 | Resize | 224×224 (EfficientNet-B4) / 299×299 (InceptionV3) |
| 4 | Convert to 3 Channels | Grayscale → Pseudo RGB for pretrained compatibility |

### 2. Data Augmentation

**Training Transformations:**

| Augmentation | Parameter |
|---|---|
| Horizontal flip | Random, p = 0.5 |
| Rotation | ±15°, uniform random |
| Random crop | Up to 10% of image border |
| Brightness adjustment | Factor from [0.8, 1.2] |
| Contrast adjustment | Factor from [0.8, 1.2] |
| Validation / Test | No augmentation applied |

**Normalization:** ImageNet channel-wise mean `µ = [0.485, 0.456, 0.406]` and std `σ = [0.229, 0.224, 0.225]`

### 3. Training Configuration

| Parameter | Value |
|---|---|
| Optimiser | AdamW |
| Initial learning rate | 3 × 10⁻⁴ |
| LR scheduler | Cosine annealing (no restart) |
| Epochs | 30 |
| Batch size | 32 |
| Loss function | Class-weighted cross-entropy |
| Class weights | [1.0, 1.0, 1.0, 1.0] |
| Label smoothing | ε = 0.1 |
| Pretrained weights | ImageNet (ILSVRC) |
| Mixed precision | Enabled (FP16) |
| Hardware | NVIDIA Tesla T4 (16 GB VRAM) |
| Approx. training time | ~2.5 hours per model |

---

## Architecture

### MSW-EnsNet (Proposed Model)

<p align="center">
  <img src="https://github.com/user-attachments/assets/c8861497-6c45-49d8-a067-048d2ff47fbd" width="900" alt="MSW-EnsNet Architecture"/>
</p>

MSW-EnsNet is a **decision-level CNN ensemble** that processes each CLAHE-enhanced chest X-ray through two parallel backbone branches and combines their class probability outputs via a **class-aware weighted fusion strategy**.

#### Mathematical Formulation

**Branch Predictions:**
```
PE = FE(Resize(I, 224×224)) ∈ R⁴       [EfficientNet-B4]
PI = FI(Resize(I, 299×299)) ∈ R⁴       [InceptionV3]
```

**Decision-Level Weighted Fusion:**
```
Pfusion = w1 · PE + w2 · PI,   w1=0.70, w2=0.30
```

**Class-Aware Posterior Reweighting:**
```
Pweighted = Pfusion ⊙ wb,   wb = [1.0, 1.0, 1.6, 1.6]
                              (COVID, Normal, Bacterial, Viral)
```

**Test-Time Augmentation (TTA) — 4 views:**
```
PTTA = (1/4) Σ P_norm(vk)
```

**Final Prediction:**
```
ŷ = argmax(PTTA,c)
```

#### TTA Views

| View | Description |
|------|-------------|
| v1 — Identity | Original CLAHE-preprocessed image |
| v2 — Horizontal flip | Left-right mirroring |
| v3 — Brightness increase | ×1.03, clamped to [0,1] |
| v4 — Brightness decrease | ×0.97, clamped to [0,1] |

#### Inference Algorithm

```
Require: Input I; pretrained FE, FI; w1=0.7, w2=0.3; wb=[1.0,1.0,1.6,1.6]
1: Apply CLAHE to I → I'
2: V ← {I', HFlip(I'), clamp(I'×1.03,0,1), clamp(I'×0.97,0,1)}
3: for each view v ∈ V:
4:     P(v)E ← FE(Resize(v, 224×224))
5:     P(v)I ← FI(Resize(v, 299×299))
6:     P(v)fus ← w1·P(v)E + w2·P(v)I
7:     P(v)wtd ← P(v)fus ⊙ wb
8:     P(v)norm ← P(v)wtd / Σc P(v)wtd,c
9: PTTA ← (1/4) Σ P(v)norm
10: ŷ ← argmax(PTTA)
```

---

## Results

### Comparative Study — All Models (Without CLAHE)

| Category | Model | Accuracy | Precision | Recall | F1-Score |
|----------|-------|----------|-----------|--------|----------|
| **CNN** | AlexNet | 0.90 | 0.90 | 0.90 | 0.90 |
| | DenseNet121 | 0.90 | 0.90 | 0.90 | 0.90 |
| | EfficientNet-B0 | 0.89 | 0.89 | 0.89 | 0.89 |
| | **EfficientNet-B4** | **0.95** | **0.95** | **0.95** | **0.95** |
| | InceptionV3 | 0.94 | 0.94 | 0.94 | 0.94 |
| | ResNet50 | 0.92 | 0.92 | 0.92 | 0.92 |
| | SqueezeNet | 0.87 | 0.87 | 0.87 | 0.87 |
| | VGG19 | 0.89 | 0.89 | 0.89 | 0.89 |
| **ViT** | BEiT | 0.88 | 0.88 | 0.88 | 0.88 |
| | Swin Transformer | 0.92 | 0.92 | 0.92 | 0.91 |
| | ViT-Base | 0.87 | 0.87 | 0.87 | 0.87 |
| **Hybrid ViT** | BoTNet | 0.88 | 0.88 | 0.88 | 0.88 |
| | ConViT | 0.92 | 0.92 | 0.92 | 0.92 |
| | MobileViT | 0.92 | 0.92 | 0.92 | 0.92 |
| | TransUNet | 0.93 | 0.93 | 0.93 | 0.93 |
| | Hybrid ResViT | 0.92 | 0.92 | 0.92 | 0.92 |
| **Ensemble** | Swin+Eff+Incep | 0.94 | 0.94 | 0.94 | 0.94 |
| | **MSW-EnsNet (Proposed)** | **0.96** | **0.96** | **0.96** | **0.96** |

### Class-Wise F1-Score Comparison

| Category | Model | COVID-19 | Normal | Bacterial | Viral |
|----------|-------|----------|--------|-----------|-------|
| CNN | AlexNet | 0.99 | 0.96 | 0.83 | 0.81 |
| | EfficientNet-B4 | 1.00 | 0.99 | 0.91 | 0.90 |
| | InceptionV3 | 1.00 | 0.98 | 0.89 | 0.89 |
| ViT | Swin Transformer | 0.99 | 0.98 | 0.85 | 0.84 |
| Hybrid | TransUNet | 1.00 | 0.98 | 0.87 | 0.86 |
| Ensemble | Swin+Eff+Incep | 0.99 | 0.98 | 0.90 | 0.89 |
| | **MSW-EnsNet** | **1.00** | **0.99** | **0.92** | **0.92** |

---

### CNN Models — Individual Results

#### AlexNet

<img src="https://github.com/user-attachments/assets/3ffe3cab-cd5e-4314-9aa1-090d7d6d9438" width="500" alt="AlexNet Results"/>
<img src="https://github.com/user-attachments/assets/81eb9370-7838-4a3d-9dd7-3af36d6b05de" width="641" alt="AlexNet Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/ce49a77f-439c-4ec0-91f0-afef68a3f531" width="695" alt="AlexNet Training Curves"/>
<img src="https://github.com/user-attachments/assets/b965a024-b900-45b7-8fe1-868f1ed299c0" width="418" alt="AlexNet ROC"/>
<img src="https://github.com/user-attachments/assets/3f72163f-44e3-4cc6-af12-2d8868c078cf" width="435" alt="AlexNet PR Curve"/>

#### DenseNet121

<img src="https://github.com/user-attachments/assets/9cf9a4bd-e1f6-4c14-a6a6-bef2e299ec73" width="562" alt="DenseNet121 Results"/>
<img src="https://github.com/user-attachments/assets/c8c4348a-947d-4245-9789-bad8ee49e4c7" width="756" alt="DenseNet121 Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/3a3eb944-5c79-4ab6-a0d3-96d230f932e3" width="898" alt="DenseNet121 Training Curves"/>
<img src="https://github.com/user-attachments/assets/2ffa0da9-883e-4698-837a-301f00634683" width="421" alt="DenseNet121 ROC"/>

#### EfficientNet-B0

<img src="https://github.com/user-attachments/assets/8bf7449e-532d-4d7f-ac81-fb4754b06ebe" width="554" alt="EfficientNetB0 Results"/>
<img src="https://github.com/user-attachments/assets/67947876-3d56-4266-ae7c-8c109dbafb81" width="662" alt="EfficientNetB0 Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/65a98919-260d-4e3e-bc73-21d78eb5c9c3" width="584" alt="EfficientNetB0 Training Curves"/>

#### EfficientNet-B4

<img src="https://github.com/user-attachments/assets/adc6cf84-c8a7-43af-9855-0ad421a40cee" width="511" alt="EfficientNetB4 Results"/>
<img src="https://github.com/user-attachments/assets/e25ff913-4fd1-4a1e-99a0-def73718c379" width="635" alt="EfficientNetB4 Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/6c96c218-d28c-4cd2-90f8-5a2408682e2b" width="578" alt="EfficientNetB4 Training Curves"/>
<img src="https://github.com/user-attachments/assets/5a0b875b-caf0-4190-bff6-f8ee8a7d9e28" width="400" alt="EfficientNetB4 ROC"/>
<img src="https://github.com/user-attachments/assets/aecb900a-9a36-4d0c-88e5-295f652c81fe" width="478" alt="EfficientNetB4 PR Curve"/>

#### InceptionV3

<img src="https://github.com/user-attachments/assets/3d310437-76c5-493d-824d-018f1b90ed6f" width="776" alt="InceptionV3 Results"/>
<img src="https://github.com/user-attachments/assets/80767082-eaf1-41a0-a313-9eb9588bc583" width="767" alt="InceptionV3 Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/d07dd67a-2660-47fa-a579-c99441ff6dfe" width="900" alt="InceptionV3 Training Curves"/>
<img src="https://github.com/user-attachments/assets/4d145c63-15ef-417c-a593-dbd10962c965" width="467" alt="InceptionV3 ROC"/>
<img src="https://github.com/user-attachments/assets/4d145c63-15ef-417c-a593-dbd10962c965" width="419" alt="InceptionV3 PR Curve"/>

#### ResNet50

<img src="https://github.com/user-attachments/assets/0145098c-60d6-4b8b-be4b-e52a57db2d55" width="530" alt="ResNet50 Results"/>
<img src="https://github.com/user-attachments/assets/780bb43a-90d0-4302-b9d9-d337dc436afe" width="666" alt="ResNet50 Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/6bab1565-80a2-4826-99f1-4283e6e280eb" width="717" alt="ResNet50 Training Curves"/>
<img src="https://github.com/user-attachments/assets/26b0796e-f5bb-4aa1-a6c5-5d5d05285b57" width="431" alt="ResNet50 ROC"/>
<img src="https://github.com/user-attachments/assets/e7a85e6e-dd0e-487c-925a-8f23e7be5070" width="466" alt="ResNet50 PR Curve"/>

#### SqueezeNet

<img src="https://github.com/user-attachments/assets/acc98369-c0d1-494d-ab5c-a41af514bbf1" width="516" alt="SqueezeNet Results"/>
<img src="https://github.com/user-attachments/assets/148f754c-014d-41a0-a64e-e7a269271b0e" width="819" alt="SqueezeNet Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/04ca6d0b-3254-4dd8-99a9-b6edec49fc1d" width="632" alt="SqueezeNet Training Curves"/>
<img src="https://github.com/user-attachments/assets/6be3c72b-bb6d-4e8e-837c-4fd83b9744d3" width="458" alt="SqueezeNet ROC"/>
<img src="https://github.com/user-attachments/assets/1a2b797a-3265-4fe0-b316-4ed81236ad9c" width="547" alt="SqueezeNet PR Curve"/>

#### VGG19

<img src="https://github.com/user-attachments/assets/556d918b-118a-42b8-b950-7120495a3256" width="573" alt="VGG19 Results"/>
<img src="https://github.com/user-attachments/assets/1f93a96f-cdba-4c76-b96e-d44b48173220" width="691" alt="VGG19 Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/1046bd1a-2c2a-4268-9d04-92cc3e3dca6b" width="536" alt="VGG19 Training Curves"/>
<img src="https://github.com/user-attachments/assets/1ce4d727-9f0a-4511-87b9-f23e8b13d3e2" width="395" alt="VGG19 ROC"/>
<img src="https://github.com/user-attachments/assets/24c42ef7-3b3e-4e09-bee9-d0c73ad400b1" width="464" alt="VGG19 PR Curve"/>

---

### Vision Transformers — Individual Results

#### Swin Transformer

<img src="https://github.com/user-attachments/assets/1a5ba23d-9b45-487a-9243-d7da253bdd9e" width="511" alt="Swin Transformer Results"/>
<img src="https://github.com/user-attachments/assets/bf7d2afe-e664-4f08-9d4a-35f446b8e8fd" width="704" alt="Swin Transformer Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/b580ba80-1722-4657-ab69-0d831593ef8b" width="584" alt="Swin Transformer Training Curves"/>
<img src="https://github.com/user-attachments/assets/0b1c28a7-1103-4ecc-bb3f-2391ff561ff7" width="379" alt="Swin Transformer ROC"/>
<img src="https://github.com/user-attachments/assets/2d6ed339-8945-4274-a958-2bcdc0fa1f69" width="378" alt="Swin Transformer PR Curve"/>

#### ViT-Base / ViT-Small-16

<img src="https://github.com/user-attachments/assets/cddb02dd-5221-41f7-a9d0-edfbe68cd863" width="569" alt="ViT Results"/>
<img src="https://github.com/user-attachments/assets/b0c74a90-d03e-4f12-9a9f-641bc087f434" width="742" alt="ViT Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/27e5bbc7-54c8-4ef2-a2ec-fc1f7da47b16" width="609" alt="ViT Training Curves"/>
<img src="https://github.com/user-attachments/assets/9bf93472-1b22-489e-85fa-15bd5f95b84e" width="444" alt="ViT ROC"/>
<img src="https://github.com/user-attachments/assets/656ec0ed-73df-4cdd-9a66-a66b6c8f9563" width="477" alt="ViT PR Curve"/>

#### BEiT

<img src="https://github.com/user-attachments/assets/e1a9f931-2121-430f-94d2-43613efc81f0" width="551" alt="BEiT Results"/>
<img src="https://github.com/user-attachments/assets/b89fbfc1-a307-41e6-96ff-b780f4f70354" width="666" alt="BEiT Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/81c8d9d8-35c5-4e2c-b05f-1b46a67c4f66" width="536" alt="BEiT Training Curves"/>
<img src="https://github.com/user-attachments/assets/dcb77e43-f909-4f80-bf1c-13f0b4ead5df" width="423" alt="BEiT ROC"/>
<img src="https://github.com/user-attachments/assets/5d1ef723-2f49-4ee2-a7f4-8aa0e4c68c22" width="447" alt="BEiT PR Curve"/>

---

### Hybrid ViT Models — Individual Results

#### BotNet

<img src="https://github.com/user-attachments/assets/5c507834-bee2-4f35-a9da-06878582065c" width="590" alt="BotNet Results"/>
<img src="https://github.com/user-attachments/assets/e16bf234-84c8-4a69-8ac8-7e95aea795c9" width="732" alt="BotNet Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/2328cfcb-b75a-4366-8d58-e3a7e0673028" width="625" alt="BotNet Training Curves"/>
<img src="https://github.com/user-attachments/assets/99aeb3dd-87ae-4ad6-a687-197872a33e59" width="439" alt="BotNet ROC"/>
<img src="https://github.com/user-attachments/assets/98f85f7b-368f-4571-b3dc-41989472b584" width="486" alt="BotNet PR Curve"/>

#### MobileViT

<img src="https://github.com/user-attachments/assets/db9fbd85-36b2-4256-b1ed-4f85bf1ccffe" width="522" alt="MobileViT Results"/>
<img src="https://github.com/user-attachments/assets/25439b81-14e2-498a-9b45-0cc72e45a0a8" width="672" alt="MobileViT Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/79d97a09-169a-4082-a84f-697e30456a16" width="544" alt="MobileViT Training Curves"/>

#### ConViT-Base

<img src="https://github.com/user-attachments/assets/f6c07c5c-c158-4e77-a5b6-301317e51dbb" width="587" alt="ConViT Results"/>
<img src="https://github.com/user-attachments/assets/7518e058-f11f-4791-8503-5187d8f0c54c" width="739" alt="ConViT Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/41141b90-e168-4636-be36-27cbea2d58db" width="601" alt="ConViT Training Curves"/>
<img src="https://github.com/user-attachments/assets/2e3ffb1d-6e45-4fad-bc31-9e28a2fe943b" width="498" alt="ConViT ROC"/>

#### TransUNet

<img src="https://github.com/user-attachments/assets/8d33c9dc-16f4-4b96-9486-32a67cb51436" width="537" alt="TransUNet Results"/>
<img src="https://github.com/user-attachments/assets/b336518a-8a87-44b0-a872-64d0b054bfa3" width="652" alt="TransUNet Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/37dd7412-9b37-4afc-93c4-c4c64c40202a" width="583" alt="TransUNet Training Curves"/>
<img src="https://github.com/user-attachments/assets/e52e2f00-9da2-4f95-b324-fe1aa7c90e2f" width="422" alt="TransUNet ROC"/>
<img src="https://github.com/user-attachments/assets/83b562e5-0fe8-4b45-9b93-c00dcf4440d6" width="485" alt="TransUNet PR Curve"/>

#### Hybrid ResViT

<img src="https://github.com/user-attachments/assets/4dd1e8c7-c024-4d0f-bc56-16580def5ed2" width="486" alt="ResViT Results"/>
<img src="https://github.com/user-attachments/assets/ed37deaf-7638-4b62-a1ad-94524b584653" width="528" alt="ResViT Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/399f398-86b2-4e51-b9d6-7d72ac39b234" width="399" alt="ResViT ROC"/>

---

### Ensemble Baseline (Swin + EfficientNet-B4 + InceptionV3)

<img src="https://github.com/user-attachments/assets/07df06c1-b305-4dae-b3a0-8fb89c936765" width="873" alt="Ensemble Baseline Results"/>
<img src="https://github.com/user-attachments/assets/a6149245-0779-4c7f-965f-a6dd7d5e4f3b" width="824" alt="Ensemble Baseline Confusion Matrix"/>

---

### MSW-EnsNet (Proposed Model — Without CLAHE)

<img src="https://github.com/user-attachments/assets/2a05c07d-505a-4c5f-af7b-152c0d8d75e0" width="576" alt="MSW-EnsNet Results"/>
<img src="https://github.com/user-attachments/assets/f868dd86-f83b-43b2-9eb4-53bcc360902e" width="719" alt="MSW-EnsNet Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/f38152f0-1ac-46d-b01b-f344fc14d5e4" width="696" alt="MSW-EnsNet Training Curves"/>

---

## Results with CLAHE Preprocessing

### EfficientNet-B4 (with CLAHE)

<img src="https://github.com/user-attachments/assets/22c73a34-1e77-48ca-a534-4c995b27d0ac" width="665" alt="EfficientNetB4 CLAHE Report"/>
<img src="https://github.com/user-attachments/assets/cc92e5be-bd53-4798-9301-fed5e01202d8" width="538" alt="EfficientNetB4 CLAHE Metrics"/>
<img src="https://github.com/user-attachments/assets/334abf06-cd11-4b2a-9a4b-fdcdc4b45380" width="696" alt="EfficientNetB4 CLAHE Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/197ff2a3-0d3a-4e82-940b-f706cf099169" width="583" alt="EfficientNetB4 CLAHE Training Curves"/>
<img src="https://github.com/user-attachments/assets/93929130-e381-4c81-a8e6-2e4b5d3acf88" width="419" alt="EfficientNetB4 CLAHE ROC"/>
<img src="https://github.com/user-attachments/assets/3b489832-4d63-4071-9478-c557c26ad83f" width="478" alt="EfficientNetB4 CLAHE PR"/>

**Classification Report (EfficientNet-B4 + CLAHE):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Covid-19 | 1.00 | 1.00 | 1.00 | 405 |
| Normal | 0.98 | 0.99 | 0.99 | 405 |
| Pneumonia-Bacterial | 0.92 | 0.89 | 0.90 | 405 |
| Pneumonia-Viral | 0.89 | 0.91 | 0.90 | 405 |
| **Accuracy** | | | **0.9475** | 1620 |

### InceptionV3 (with CLAHE)

<img src="https://github.com/user-attachments/assets/b369db0a-f45e-48b3-b462-ab29b9cc0e08" width="563" alt="InceptionV3 CLAHE Report"/>
<img src="https://github.com/user-attachments/assets/423f6c25-4e43-4a94-bc9d-d22716b8fc5e" width="292" alt="InceptionV3 CLAHE Metrics"/>
<img src="https://github.com/user-attachments/assets/6f5867c3-04ed-4459-997f-8537eae1a396" width="743" alt="InceptionV3 CLAHE Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/f92199b6-d697-4dc9-81d1-43265fd8c8e4" width="645" alt="InceptionV3 CLAHE Training Curves"/>
<img src="https://github.com/user-attachments/assets/8612a9db-c137-4ab3-8af7-a099387c67e3" width="456" alt="InceptionV3 CLAHE ROC"/>
<img src="https://github.com/user-attachments/assets/c4234990-acad-4e9b-9ce2-4b2c0068c31e" width="460" alt="InceptionV3 CLAHE PR"/>

**Classification Report (InceptionV3 + CLAHE):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Covid-19 | 1.00 | 1.00 | 1.00 | 405 |
| Normal | 0.99 | 0.99 | 0.99 | 405 |
| Pneumonia-Bacterial | 0.89 | 0.88 | 0.89 | 405 |
| Pneumonia-Viral | 0.89 | 0.89 | 0.89 | 405 |
| **Accuracy** | | | **0.9407** | 1620 |

### MSW-EnsNet (with CLAHE) — Full Proposed Model

<img src="https://github.com/user-attachments/assets/218b27d2-a501-4d3e-9ce9-3e3dba97b9c8" width="366" alt="MSW-EnsNet CLAHE Report 1"/>
<img src="https://github.com/user-attachments/assets/ebbcc7a0-18e3-4b49-a8e4-328838c9333f" width="373" alt="MSW-EnsNet CLAHE Report 2"/>
<!-- <img src="https://github.com/user-attachments/assets/47c7b4b5-6573-4827-b2a4-b5f54e6e170c" width="582" alt="MSW-EnsNet CLAHE Metrics"/> -->

<!-- <img src="https://github.com/user-attachments/assets/6fb8eacd-f8c5-4bac-8002-74e529a0814e" width="318" alt="MSW-EnsNet CLAHE Analysis 5"/>/> -->

<img src="https://github.com/user-attachments/assets/7b69a539-1c96-4228-a92e-1d66a42adebe" width="593" alt="MSW-EnsNet CLAHE Training Curves"/>
<img src="https://github.com/user-attachments/assets/0698b363-a4e5-431a-b849-c75aeee36988" width="636" alt="MSW-EnsNet CLAHE Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/b6fad742-1af4-4abf-ad6b-6649c9683f05" width="625" alt="MSW-EnsNet CLAHE ROC"/>
<img src="https://github.com/user-attachments/assets/b0b2e132-017e-490a-912b-80864cf790c0" width="404" alt="MSW-EnsNet CLAHE Analysis 1"/>
<img src="https://github.com/user-attachments/assets/143f255c-df03-447d-9371-01f8c6661a7f" width="516" alt="MSW-EnsNet CLAHE Analysis 2"/>
<img src="https://github.com/user-attachments/assets/541f287c-a33e-4d86-9cf2-5e62f9793977" width="541" alt="MSW-EnsNet CLAHE Analysis 3"/>
<img src="https://github.com/user-attachments/assets/562f383c-c5a5-41ef-99ad-fda2bfb1abc3" width="562" alt="MSW-EnsNet CLAHE Analysis 4"/>

<img src="https://github.com/user-attachments/assets/7a4bc969-dfab-459f-96f4-ad665e18d8b0" width="537" alt="MSW-EnsNet CLAHE Analysis 6"/>
<img src="https://github.com/user-attachments/assets/45ba1e38-20aa-4133-a2c7-af6ca7d22d46" width="392" alt="MSW-EnsNet CLAHE Analysis 7"/>
<img src="https://github.com/user-attachments/assets/90772485-e9d5-4dc7-a11a-479621d053db" width="533" alt="MSW-EnsNet CLAHE Analysis 8"/>

**Classification Report (MSW-EnsNet + CLAHE):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Covid-19 | 0.99 | 1.00 | 1.00 | 405 |
| Normal | 0.99 | 0.98 | 0.99 | 405 |
| Pneumonia-Bacterial | 0.93 | 0.91 | 0.92 | 405 |
| Pneumonia-Viral | 0.91 | 0.93 | 0.92 | 405 |
| **Accuracy** | | | **0.9543** | 1620 |
| **Macro Avg** | **0.9550** | **0.9550** | **0.9544** | 1620 |

**Confusion Matrix (MSW-EnsNet):**

| Actual \ Predicted | COVID | Normal | Bacterial | Viral |
|-------------------|-------|--------|-----------|-------|
| COVID-19 | **404** | 0 | 0 | 1 |
| Normal | 0 | **398** | 0 | 7 |
| Bact. Pneumonia | 1 | 2 | **363** | 39 |
| Viral Pneumonia | 0 | 4 | 21 | **380** |

### TransUNet (with CLAHE)

<img src="https://github.com/user-attachments/assets/c278862e-94f9-49f6-87d7-8199d4f72008" width="520" alt="TransUNet CLAHE Results"/>
<img src="https://github.com/user-attachments/assets/204da2c1-cff0-4a47-8918-9154b02c9bff" width="629" alt="TransUNet CLAHE Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/63b23aee-1351-4ab3-9685-8a0473cf13ed" width="632" alt="TransUNet CLAHE Metrics"/>
<img src="https://github.com/user-attachments/assets/73f502a1-83b7-4c21-9663-6cc49d3c42db" width="614" alt="TransUNet CLAHE Training"/>

### ResViT (with CLAHE)

<img src="https://github.com/user-attachments/assets/56e9c5cf-05dd-4977-80ea-34736a35efe5" width="313" alt="ResViT CLAHE Results"/>
<img src="https://github.com/user-attachments/assets/295f4731-6fda-4de4-addb-5232b9732e59" width="579" alt="ResViT CLAHE Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/462f428-1e6b-4db3-bb02-e0d12d1c6b35" width="462" alt="ResViT CLAHE ROC"/>
<img src="https://github.com/user-attachments/assets/2f59f8b5-de20-4620-b896-70868fe9685a" width="252" alt="ResViT CLAHE Metrics"/>

### Swin Transformer (with CLAHE)

<img src="https://github.com/user-attachments/assets/a0ecc640-e534-4ade-8d44-34db77f3183b" width="519" alt="Swin CLAHE Results"/>
<img src="https://github.com/user-attachments/assets/4cc1f166-4829-414d-a6e7-a24f935ed51a" width="646" alt="Swin CLAHE Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/fc08a721-5010-4f6c-a7ea-38e9287bb5ec" width="559" alt="Swin CLAHE Training"/>
<img src="https://github.com/user-attachments/assets/e4e69112-16ac-4970-91df-9b4a309efd03" width="476" alt="Swin CLAHE ROC"/>
<img src="https://github.com/user-attachments/assets/329f175" width="329" alt="Swin CLAHE PR"/>

### Ensemble (Swin + EfficientNet-B4 + InceptionV3) with CLAHE

<img src="https://github.com/user-attachments/assets/af87b19b-f4d8-4315-8035-abb90733d133" width="617" alt="Ensemble CLAHE Results"/>
<img src="https://github.com/user-attachments/assets/83c1a49a-c316-4a99-8c8b-e6f1959cf211" width="735" alt="Ensemble CLAHE Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/fa081c22-e3dd-4134-9223-976dd7c619d1" width="412" alt="Ensemble CLAHE Metrics"/>

---

## Extended Evaluation Metrics

| Model | Accuracy | Macro F1 | Bal. Acc. | AUC-ROC | Cohen's κ | 95% CI (F1) |
|-------|----------|----------|-----------|---------|-----------|-------------|
| EfficientNet-B4 | 0.9475 | 0.9474 | 0.9475 | 0.9771 | 0.9300 | (0.936, 0.957) |
| InceptionV3 | 0.9407 | 0.9406 | 0.9407 | 0.9918 | 0.9210 | (0.929, 0.951) |
| Swin Transformer | 0.9111 | 0.9112 | 0.9111 | 0.9860 | 0.8815 | (0.897, 0.925) |
| Ensemble (Swin+Eff+Inc) | 0.9432 | 0.9429 | 0.9432 | 0.9894 | 0.9243 | (0.932, 0.955) |
| **MSW-EnsNet** | **0.9543** | **0.9544** | **0.9543** | **0.9913** | **0.9391** | **(0.944, 0.965)** |

### Statistical Significance (McNemar's Test, α = 0.05)

| Model | Macro F1 | 95% CI | p-value |
|-------|----------|--------|---------|
| EfficientNet-B4 | 0.9474 | (0.936, 0.957) | 0.0148 |
| InceptionV3 | 0.9406 | (0.929, 0.951) | 4.05 × 10⁻²¹ |
| Swin Transformer | 0.9112 | (0.897, 0.925) | < 0.0001 |
| **MSW-EnsNet** | **0.9544** | **(0.944, 0.965)** | — |

---

## Ablation Study

### Component Contribution

| Variant | Macro F1 | Accuracy |
|---------|----------|----------|
| EfficientNet-B4 only | 0.9474 | 0.9475 |
| InceptionV3 only | 0.9406 | 0.9407 |
| Simple average ensemble | 0.9449 | 0.9450 |
| Weighted fusion, no class boost | 0.9481 | 0.9481 |
| Weighted fusion + class boost, no TTA | 0.9463 | 0.9463 |
| Full model without TTA | 0.9481 | 0.9481 |
| Full model without CLAHE | 0.9420 | 0.9420 |
| Full model without label smoothing | 0.9423 | 0.9426 |
| **MSW-EnsNet (full proposed)** | **0.9544** | **0.9543** |

### Effect of Class-Aware Boost Factor

| Boost (Bac./Vir.) | Macro F1 | Accuracy |
|-------------------|----------|----------|
| 1.0 | 0.9512 | 0.9512 |
| 1.2 | 0.9519 | 0.9519 |
| 1.4 | 0.9531 | 0.9531 |
| **1.6** | **0.9544** | **0.9543** |
| 1.8 | 0.9544 | 0.9543 |
| 2.0 | 0.9544 | 0.9543 |

> Optimal boost = **1.6** (first value at which performance plateaus)

### Effect of Fusion Weight

| w1 (EfficientNet-B4) | w2 (InceptionV3) | Macro F1 | Accuracy |
|----------------------|------------------|----------|----------|
| 0.3 | 0.7 | 0.9494 | 0.9494 |
| 0.5 | 0.5 | 0.9526 | 0.9525 |
| 0.6 | 0.4 | 0.9532 | 0.9531 |
| **0.7** | **0.3** | **0.9544** | **0.9543** |
| 0.8 | 0.2 | 0.9512 | 0.9512 |
| 1.0 | 0.0 | 0.9488 | 0.9488 |

### Ablation Results

#### Without CLAHE (Proposed Model)

<img src="https://github.com/user-attachments/assets/e0e02814-1a43-40c4-9869-21e131295367" width="844" alt="MSW-EnsNet Without CLAHE Results"/>

#### Without Label Smoothing (Proposed Model)

<img src="https://github.com/user-attachments/assets/207b9b90-a6a4-4b95-8fe6-8a61138e1eab" width="768" alt="MSW-EnsNet Without Label Smoothing Results"/>

---

## Explainability

### Grad-CAM Saliency Maps

Grad-CAM saliency maps were generated by computing the gradient of the predicted class score with respect to the final convolutional feature maps of EfficientNet-B4. Activation is concentrated in pathologically relevant lung regions:

- **COVID-19**: Bilateral lower and peripheral lung zones
- **Bacterial pneumonia**: Lobar consolidation regions
- **Normal**: Diffuse, low-intensity activation (no spurious attention)

<img src="https://github.com/user-attachments/assets/f7c87c0b-e38f-4a48-92e0-c4ae49bc3671" width="851" alt="Grad-CAM Results"/>

### RISE Pixel Importance Maps

RISE maps (2,000 random binary masks per image) confirmed that:
- High-importance regions = pathologically significant lung areas (infiltrates, consolidations, ground-glass opacities)
- Background regions = consistently low importance scores
- Imaging artefacts, bony thorax, patient identifiers = not attended to

<img src="https://github.com/user-attachments/assets/3bcf3b51-7bc8-4928-a18d-25314f1aa333" width="819" alt="RISE Pixel Importance Maps"/>

---

## Computational Complexity

| Model | Params (M) | FLOPs (G) | Size (MB) | Infer. Time (ms/img) | GPU Mem (GB) | Macro F1 |
|-------|-----------|-----------|-----------|---------------------|--------------|----------|
| EfficientNet-B4 | 19.34 | 1.58 | 74.65 | 15.24 | 0.59 | 0.9474 |
| InceptionV3 | 23.83 | 2.86 | 91.27 | 11.03 | 0.11 | 0.9406 |
| Swin Transformer | 28.29 | 2.98 | 108.22 | 12.58 | 0.22 | 0.9112 |
| TransUNet | 60.86 | 8.90 | 232.51 | 11.12 | 0.52 | 0.8975 |
| DCNN-ViT-GRU | 110.75 | 15.42 | 422.87 | 20.13 | 0.54 | 0.9148 |
| **MSW-EnsNet** | **43.17** | **4.44** | **165.92** | **27.41** | **0.70** | **0.9544** |

> MSW-EnsNet achieves higher macro F1 than DCNN-ViT-GRU with **74% fewer parameters** and **71% fewer FLOPs**.

---

## External Validation

### Zero-Shot Transfer Results

| Train Set | External Dataset | Task | Accuracy | Macro F1 |
|-----------|-----------------|------|----------|----------|
| ChestX6 | Kermany | Normal vs. Pneumonia | 0.8352 | 0.8066 |
| ChestX6 | COVID-19 Radiography DB | COVID-19 vs. Normal | 0.6739 | 0.5693 |

### Kermany Dataset Validation

<img src="https://github.com/user-attachments/assets/f7c87c0b-e38f-4a48-92e0-c4ae49bc3671" width="851" alt="Kermany Dataset Validation"/>
<img src="https://github.com/user-attachments/assets/3bcf3b51-7bc8-4928-a18d-25314f1aa333" width="819" alt="Kermany Dataset Detailed Results"/>

### COVID-19 Radiography Database Validation

<img src="https://github.com/user-attachments/assets/6c4093ce-bee3-4de2-87bf-ced88164d7c9" width="831" alt="COVID-19 Radiography DB Validation"/>

### Fine-Tuning on Kermany

| Setting | Accuracy | Macro F1 |
|---------|----------|----------|
| Before fine-tuning | 0.8352 | 0.8066 |
| After fine-tuning (10–20% data, classifier only) | 0.8777 | 0.8761 |

---

## Limitations

- **Single-dataset training**: Trained exclusively on ChestX6 (Kaggle); external multi-centre validation is required.
- **No patient-level identifiers**: Same patient may appear in both train and test sets — a known limitation of public datasets.
- **Explainability ≠ clinical validation**: Grad-CAM and RISE provide visual evidence but not formal radiological validation.
- **Dual-backbone overhead**: Higher inference time (27.41 ms) vs. single models — model compression via pruning/distillation is a future direction.
- **Screening tool only**: All predictions must be reviewed by a qualified clinician before clinical action.

---

## Future Work

- External validation on multi-centre datasets with patient-level provenance
- Patient-level splitting and near-duplicate image detection
- Model compression via structured pruning and knowledge distillation
- Bayesian uncertainty estimation for confidence-aware predictions
- Adaptive attention-based backbone fusion weights
- Lung segmentation as preprocessing to constrain model attention
- Radiologist-in-the-loop structured evaluation
- Prospective clinical validation

---

## Installation & Usage

### Requirements

```bash
pip install torch torchvision timm opencv-python scikit-learn numpy matplotlib
```

### Quick Start

```python
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# Load CLAHE preprocessing
def apply_clahe(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    return img_rgb

# Transform for EfficientNet-B4
transform_eff = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Transform for InceptionV3
transform_inc = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# TTA views
def get_tta_views(image_tensor):
    t1 = image_tensor                                      # Original
    t2 = torch.flip(image_tensor, dims=[3])                # Horizontal flip
    t3 = torch.clamp(image_tensor * 1.03, 0, 1)           # Brightness +
    t4 = torch.clamp(image_tensor * 0.97, 0, 1)           # Brightness -
    return [t1, t2, t3, t4]

# MSW-EnsNet inference
def msw_ensnet_predict(image_path, eff_model, inc_model):
    classes = ['COVID-19', 'Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']
    w1, w2 = 0.70, 0.30
    wb = torch.tensor([1.0, 1.0, 1.6, 1.6])
    
    img_clahe = apply_clahe(image_path)
    img_pil = Image.fromarray(img_clahe)
    
    img_eff = transform_eff(img_pil).unsqueeze(0)
    img_inc = transform_inc(img_pil).unsqueeze(0)
    
    tta_views_eff = get_tta_views(img_eff)
    tta_views_inc = get_tta_views(img_inc)
    
    ptta = torch.zeros(4)
    for ve, vi in zip(tta_views_eff, tta_views_inc):
        pe = torch.softmax(eff_model(ve), dim=1).squeeze()
        pi = torch.softmax(inc_model(vi), dim=1).squeeze()
        pfus = w1 * pe + w2 * pi
        pwtd = pfus * wb
        pnorm = pwtd / pwtd.sum()
        ptta += pnorm
    
    ptta /= 4
    pred_class = classes[ptta.argmax().item()]
    return pred_class, ptta
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{kondra2025mswensnet,
  title={MSW-EnsNet: A Class-Aware Weighted CNN Ensemble with Test-Time Augmentation for Multi-Class Pneumonia Classification from Chest X-rays},
  author={Kondra, Venkat Sai and Kumar, Chintha Sampath and Reddy, Anudeep},
  institution={VNR Vignana Jyothi Institute of Engineering and Technology},
  year={2025}
}
```

---

## Authors

| Name | Institution | Email |
|------|-------------|-------|
| Venkat Sai Kondra | VNR VJIET, Hyderabad | venkatsaikondra67@gmail.com |
| Chintha Sampath Kumar | VNR VJIET, Hyderabad | chinthasampath3763@gmail.com |
| Anudeep Reddy | VNR VJIET, Hyderabad | cherukuanudeepreddy@gmail.com |

---

<p align="center">
  <i>This model is intended as a clinical decision-support screening tool only. All predictions must be reviewed by a qualified clinician before any clinical action is taken.</i>
</p>




readme.txt

ablation with validation sd and final results

<img width="774" height="276" alt="image" src="https://github.com/user-attachments/assets/ff0de305-4599-414d-bb33-0f48341b0d28" />

<img width="740" height="325" alt="image" src="https://github.com/user-attachments/assets/8d57bd2c-2d69-4695-8d16-32655460efbb" />
<img width="796" height="89" alt="image" src="https://github.com/user-attachments/assets/73b44e49-d0c3-4a8a-93f7-021e7581f8ca" />

<img width="576" height="312" alt="image" src="https://github.com/user-attachments/assets/b33204a9-8174-4462-b496-1ed9acb2aefa" />


<img width="537" height="669" alt="image" src="https://github.com/user-attachments/assets/e13a79c2-80a4-40fe-a9fa-b28a366d0e7c" />

<img width="560" height="77" alt="image" src="https://github.com/user-attachments/assets/15468048-022c-4b8a-8ce5-1fb500154a70" />

<img width="668" height="118" alt="image" src="https://github.com/user-attachments/assets/8f7acaaf-420c-4776-8eb2-ebddc55b2db1" />

<img width="688" height="294" alt="image" src="https://github.com/user-attachments/assets/a0cdd9a8-6580-4668-bae4-02a57e6736ef" />

<img width="669" height="646" alt="image" src="https://github.com/user-attachments/assets/dea0fc4e-ea72-45f8-93bd-644f7ebfe46c" />

<img width="529" height="348" alt="image" src="https://github.com/user-attachments/assets/b33644b8-1c3c-40a3-9822-e5bb47f9981f" />

<img width="644" height="647" alt="image" src="https://github.com/user-attachments/assets/e5a07882-e742-49b9-b586-e53331fd095f" />

<img width="499" height="359" alt="image" src="https://github.com/user-attachments/assets/ae4bb65a-1730-4611-9fbb-0a8e6a86a3b0" />


<img width="734" height="504" alt="image" src="https://github.com/user-attachments/assets/473e910f-f8ae-454e-818e-d1b8293a8bd1" />

<img width="592" height="134" alt="image" src="https://github.com/user-attachments/assets/e565f3db-b963-4b55-86fb-a0b9e1e8e708" />
