# Pneumonia_Research

<img width="234" height="167" alt="image" src="https://github.com/user-attachments/assets/dd123697-f9f5-4c7f-8cf6-eba83fe645c2" />

># 🫁 “Multi-Scale Weighted Ensemble of CNN Architectures with Test-Time Augmentation for Robust Multi-Class Pneumonia Classification”

> A systematic comparative study of Convolutional Neural Networks, Vision Transformers, and a Hybrid-ViT architecture on balanced, separately-augmented chest X-ray data for multi-class pneumonia detection.

Introduction
------------

Pneumonia, a respiratory infection caused by either viral or bacterial pathogens.In humans, the lungs contain tiny sacs called alveoli, which fill with air during healthy breathing.When pneumonia occurs, thes aveoli become filled with pus and fluid, causing discomfory and difficulty in breathing and restricting oxygen intake.Statistically, pneumonia impacts approximately 1,400 children per 100,000,contributing to 14% of fatalities among children below five years of age in 2019, resulting in the deaths of 740,180 children. In 2017, pneumonia caused over 808,000 deaths among children under five, representing 15% of all fatalities in this age group.The reduction in these figures can be
attributed to improved detection techniques and the development of health technologies such as vaccines.
The primary diagnostic tool for pneumonia is chest X-rays,where areas of infection are identified by white patches in the pulmonary region.

CNN:-
The CNN model examines connections between neighboring pixels within a specific receptive field defined by the filter size. Despite its effectiveness in capturing local spatial hierarchies, it struggles with recognizing relationships between distant pixels, which hinders its ability to understand the broader context and more complex patterns within an image. 
To overcome this limitation, recent advancements have integrated attention mechanisms, allowing the model to dynamically weigh the importance of pixels, regardless of their distance from each other. This enhancement helps capture both local and global dependencies more effectively.

Vision Transformer (ViT):-
Unlike CNNs, which are restricted by the local receptive field of filters, the Vision Transformer (ViT) architecture abandons convolutions entirely in favor of Self-Attention mechanisms. It treats an image as a sequence of patches (visual tokens), allowing every pixel to interact with every other pixel across the entire image simultaneously.

Hybrid-ViT :-
The Hybrid-ViT architecture represents the current state-of-the-art for your pneumonia classification task. It combines the best of both worlds: using a CNN backbone (like ResNet or BiT) to extract low-level spatial features (textures and edges) and Transformer blocks to process those features with global attention.

## 📌 Overview

There are 4 classes of dataset that had taken.

| Class | Description |
|-------|-------------|
| `Normal` | Healthy chest X-ray |
| `Viral Pneumonia` | Pneumonia caused by viral infection |
| `Bacterial Pneumonia` | Pneumonia caused by bacterial infection |
| `COVID-19 Pneumonia` | Pneumonia associated with SARS-CoV-2 |

## Methodology

Data Collection
---------------

From Kaggle

The data was took from kaggle(https://www.kaggle.com/datasets/mohamedasak/chest-x-ray-6-classes-dataset)
After adding all the data it became as 
#Covid-2717
#Normal-2971
#Bacterial-2699
#Viral-2713

1. Class Balancing
******************
To eliminate class imbalance and avoid biased model learning:

-> The class with the minimum samples (2699) was selected as the reference.

->All other classes were downsampled to 2699 images.

4 classes × 2699 images each = 10,784 total images

Data Preprocessing
--------------------

2. Image Preprocessing
***********************

Each image undergoes preprocessing using OpenCV before feeding into the model.

   Step 1: Load Image in Grayscale

-> Converts chest X-ray into single-channel grayscale.

-> Removes unnecessary color information (important for medical images)

   Step 2: Apply CLAHE (Contrast Enhancement)

-> CLAHE = Contrast Limited Adaptive Histogram Equalization

-> Enhances local contrast

-> Highlights important lung features (e.g., opacities, infections)

   Step 3: Resize Image

-> Standardizes all images to 224 × 224

-> Required for models like ResNet, ViT, EfficientNet

   Step 4: Convert to 3 Channels

-> Converts grayscale → pseudo RGB (3 channels)

-> Needed because pretrained models expect 3-channel input


3. Data Augmentation & Normalization
*************************************

Training Transformations
---------------------------

Resize → ensures uniform input size
RandomHorizontalFlip → improves generalization
RandomRotation (±10°) → simulates real-world variation
ColorJitter → adds slight intensity variation
ToTensor → converts image to PyTorch tensor
Normalization → scales pixel values to [-1, 1]

Validation & Test Transformations
----------------------------------

No augmentation applied

Ensures fair evaluation

4. Dataset Splitting
*********************

The balanced dataset (Equal_sample/) is split into:

-> Training set (70%)
-> Validation set (15%)
-> Test set (15%)

Random splitting ensures unbiased distribution

random_state=42 ensures reproducibility

5. Directory Structure Creation
__________________________________

The processed dataset is organized into:

Final_Data/
    ├── train/
    ├── val/
    └── test/
        ├── Covid-19/
        ├── Normal/
        ├── Pneumonia-Bacterial/
        └── Pneumonia-Viral/

CNN
********

Alexnet:-

<img width="833" height="212" alt="image" src="https://github.com/user-attachments/assets/55bbf428-d413-48de-a27e-24e221daed21" />

<img width="844" height="676" alt="image" src="https://github.com/user-attachments/assets/346b1b09-e37f-4004-921b-67b076a9b791" />

DenseNet121:-

<img width="829" height="214" alt="image" src="https://github.com/user-attachments/assets/ffb21695-a06e-4f40-863a-bc650ed303f7" />

<img width="823" height="682" alt="image" src="https://github.com/user-attachments/assets/acb6fa09-486e-42e1-9316-aca771ae8807" />

EfficientB0

<img width="837" height="223" alt="image" src="https://github.com/user-attachments/assets/40aaf6eb-1cb7-4195-8518-ccbe9bb7c88a" />

<img width="876" height="683" alt="image" src="https://github.com/user-attachments/assets/09163e33-6932-4f10-a731-c13147ff2acd" />

InceptionV3

<img width="776" height="244" alt="image" src="https://github.com/user-attachments/assets/3d310437-76c5-493d-824d-018f1b90ed6f" />


<img width="829" height="685" alt="image" src="https://github.com/user-attachments/assets/8f93d702-d817-41a4-a429-6d3b3deb3d29" />

ResNet50

<img width="751" height="244" alt="image" src="https://github.com/user-attachments/assets/9306fd4d-6057-4575-ab36-9133d4822db8" />

<img width="822" height="684" alt="image" src="https://github.com/user-attachments/assets/be5b8531-8d05-496c-a3c9-2538b67f4b0b" />

SqueezeNet

<img width="761" height="261" alt="image" src="https://github.com/user-attachments/assets/00415a03-86cd-4bb1-80b9-dbe8af2bc072" />

<img width="829" height="682" alt="image" src="https://github.com/user-attachments/assets/368857bd-e1cb-45b3-b77d-404514c9d966" />

VGG19

<img width="842" height="240" alt="image" src="https://github.com/user-attachments/assets/ebce4667-7f38-4346-a35a-6aa05a804af4" />

<img width="867" height="690" alt="image" src="https://github.com/user-attachments/assets/e37b4013-5cd7-41e1-a2ec-9e5ef5430b97" />

Vision_Transformers

Swim_Transformer

<img width="838" height="218" alt="image" src="https://github.com/user-attachments/assets/31bfd1f8-edcc-40d6-b497-19dc01506566" />

<img width="837" height="678" alt="image" src="https://github.com/user-attachments/assets/28f91351-daa6-4afe-93d0-6ae75e14b6b2" />

Vit_Base_Small_16

<img width="1223" height="238" alt="image" src="https://github.com/user-attachments/assets/0a9ed8db-a314-4e15-bb20-e9a18fa815e0" />

<img width="1002" height="691" alt="image" src="https://github.com/user-attachments/assets/0c84be45-e3aa-40d5-9769-492290fd8244" />


BEiT

<img width="1064" height="244" alt="image" src="https://github.com/user-attachments/assets/c080b850-7044-418a-9906-df70e4cba8eb" />

<img width="835" height="683" alt="image" src="https://github.com/user-attachments/assets/7a6aff63-b909-438a-93a1-9708de4c6c45" />

Hybrid_ViT
-----------

BotNet

<img width="1081" height="209" alt="image" src="https://github.com/user-attachments/assets/479e7131-503c-477c-8327-0c223b9a95aa" />

<img width="1057" height="685" alt="image" src="https://github.com/user-attachments/assets/6d0c2416-4d62-4c71-aa30-a34fb5cef148" />

Mobile Vit

<img width="809" height="229" alt="image" src="https://github.com/user-attachments/assets/826cdca8-2a4d-4ed2-ab01-f87f77e3f830" />

<img width="979" height="694" alt="image" src="https://github.com/user-attachments/assets/68ee439a-c9e1-461a-96fd-8a80083e8f8a" />


ConViT

<img width="1032" height="227" alt="image" src="https://github.com/user-attachments/assets/eb700267-7426-4363-8fae-dcdae9281b81" />

<img width="956" height="694" alt="image" src="https://github.com/user-attachments/assets/e1f26754-1cd3-4464-a696-103811e91bd8" />

TransUNet

<img width="1167" height="211" alt="image" src="https://github.com/user-attachments/assets/0b2a3897-526c-482e-bcf4-6c6e33bbe5bd" />

<img width="1207" height="674" alt="image" src="https://github.com/user-attachments/assets/fe10ce06-310e-4f2c-8ca6-36421f9ec42a" />

Hybrid_ResViT

Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.99      1.00       404
           1       0.98      0.98      0.98       404
           2       0.84      0.87      0.85       404
           3       0.86      0.83      0.85       404

    accuracy                           0.92      1616
    
   macro avg       0.92      0.92      0.92      1616
   
weighted avg       0.92      0.92      0.92      1616

Confusion Matrix:

[[401   1   0   2]

 [  0 395   5   4]
 
 [  0   3 353  48]
 
 [  0   3  64 337]]
 
Epoch 20,  Loss: 195.2859,   Val Acc: 0.9196


Proposed_Model
+++++++++++++++

MSW-EnsNet

<img width="869" height="236" alt="image" src="https://github.com/user-attachments/assets/744e4cfa-ab43-4ac8-bbd8-5cdf94b862d1" />


<img width="881" height="438" alt="image" src="https://github.com/user-attachments/assets/36bb8308-6fbb-4053-a322-2791691f75f0" />



<img width="843" height="460" alt="image" src="https://github.com/user-attachments/assets/9596e214-2c4f-433b-b5e2-8653be573d3f" />



<img width="872" height="617" alt="image" src="https://github.com/user-attachments/assets/f545d01b-209a-4303-84ec-4b7129cee960" />


Analysis
++++++++


<img width="845" height="652" alt="image" src="https://github.com/user-attachments/assets/19dbbcae-c6e3-41e0-a977-e1347ed62890" />












