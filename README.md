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


<img width="500" height="198" alt="image" src="https://github.com/user-attachments/assets/3ffe3cab-cd5e-4314-9aa1-090d7d6d9438" />

<img width="641" height="623" alt="image" src="https://github.com/user-attachments/assets/81eb9370-7838-4a3d-9dd7-3af36d6b05de" />


<img width="695" height="513" alt="image" src="https://github.com/user-attachments/assets/ce49a77f-439c-4ec0-91f0-afef68a3f531" />


<img width="418" height="412" alt="image" src="https://github.com/user-attachments/assets/b965a024-b900-45b7-8fe1-868f1ed299c0" />


<img width="435" height="419" alt="image" src="https://github.com/user-attachments/assets/3f72163f-44e3-4cc6-af12-2d8868c078cf" />


DenseNet121:-

<img width="562" height="227" alt="image" src="https://github.com/user-attachments/assets/9cf9a4bd-e1f6-4c14-a6a6-bef2e299ec73" />


<img width="756" height="690" alt="image" src="https://github.com/user-attachments/assets/c8c4348a-947d-4245-9789-bad8ee49e4c7" />

<img width="898" height="699" alt="image" src="https://github.com/user-attachments/assets/3a3eb944-5c79-4ab6-a0d3-96d230f932e3" />



<img width="421" height="434" alt="image" src="https://github.com/user-attachments/assets/2ffa0da9-883e-4698-837a-301f00634683" />


EfficientNetB0

<img width="554" height="215" alt="image" src="https://github.com/user-attachments/assets/8bf7449e-532d-4d7f-ac81-fb4754b06ebe" />

<img width="662" height="614" alt="image" src="https://github.com/user-attachments/assets/67947876-3d56-4266-ae7c-8c109dbafb81" />

<img width="584" height="444" alt="image" src="https://github.com/user-attachments/assets/65a98919-260d-4e3e-bc73-21d78eb5c9c3" />


EfficientNetB4

<img width="709" height="270" alt="image" src="https://github.com/user-attachments/assets/915a7136-8bbf-4a85-8582-e3d742c4ef59" />

<img width="870" height="632" alt="image" src="https://github.com/user-attachments/assets/2ba58913-6ecd-432e-a05d-9e344b044c50" />


InceptionV3

<img width="776" height="244" alt="image" src="https://github.com/user-attachments/assets/3d310437-76c5-493d-824d-018f1b90ed6f" />


<img width="767" height="700" alt="image" src="https://github.com/user-attachments/assets/80767082-eaf1-41a0-a313-9eb9588bc583" />



<img width="1394" height="944" alt="image" src="https://github.com/user-attachments/assets/d07dd67a-2660-47fa-a579-c99441ff6dfe" />


<img width="467" height="424" alt="image" src="https://github.com/user-attachments/assets/4d145c63-15ef-417c-a593-dbd10962c965" />


<img width="419" height="416" alt="image" src="https://github.com/user-attachments/assets/c53abfb8-94be-468a-860f-1e2d8fd318e0" />



ResNet50

<img width="530" height="197" alt="image" src="https://github.com/user-attachments/assets/0145098c-60d6-4b8b-be4b-e52a57db2d55" />


<img width="666" height="626" alt="image" src="https://github.com/user-attachments/assets/780bb43a-90d0-4302-b9d9-d337dc436afe" />

<img width="717" height="515" alt="image" src="https://github.com/user-attachments/assets/6bab1565-80a2-4826-99f1-4283e6e280eb" />


<img width="431" height="393" alt="image" src="https://github.com/user-attachments/assets/26b0796e-f5bb-4aa1-a6c5-5d5d05285b57" />

<img width="466" height="380" alt="image" src="https://github.com/user-attachments/assets/e7a85e6e-dd0e-487c-925a-8f23e7be5070" />


SqueezeNet

<img width="516" height="207" alt="image" src="https://github.com/user-attachments/assets/acc98369-c0d1-494d-ab5c-a41af514bbf1" />

<img width="819" height="623" alt="image" src="https://github.com/user-attachments/assets/148f754c-014d-41a0-a64e-e7a269271b0e" />

<img width="632" height="423" alt="image" src="https://github.com/user-attachments/assets/04ca6d0b-3254-4dd8-99a9-b6edec49fc1d" />

<img width="458" height="409" alt="image" src="https://github.com/user-attachments/assets/6be3c72b-bb6d-4e8e-837c-4fd83b9744d3" />

<img width="547" height="393" alt="image" src="https://github.com/user-attachments/assets/1a2b797a-3265-4fe0-b316-4ed81236ad9c" />


VGG19

<img width="573" height="216" alt="image" src="https://github.com/user-attachments/assets/556d918b-118a-42b8-b950-7120495a3256" />

<img width="691" height="627" alt="image" src="https://github.com/user-attachments/assets/1f93a96f-cdba-4c76-b96e-d44b48173220" />

<img width="536" height="438" alt="image" src="https://github.com/user-attachments/assets/1046bd1a-2c2a-4268-9d04-92cc3e3dca6b" />

<img width="395" height="405" alt="image" src="https://github.com/user-attachments/assets/1ce4d727-9f0a-4511-87b9-f23e8b13d3e2" />

<img width="464" height="438" alt="image" src="https://github.com/user-attachments/assets/24c42ef7-3b3e-4e09-bee9-d0c73ad400b1" />

Vision_Transformers

Swim_Transformer

<img width="511" height="209" alt="image" src="https://github.com/user-attachments/assets/1a5ba23d-9b45-487a-9243-d7da253bdd9e" />

<img width="704" height="633" alt="image" src="https://github.com/user-attachments/assets/bf7d2afe-e664-4f08-9d4a-35f446b8e8fd" />

<img width="584" height="427" alt="image" src="https://github.com/user-attachments/assets/b580ba80-1722-4657-ab69-0d831593ef8b" />

<img width="379" height="397" alt="image" src="https://github.com/user-attachments/assets/0b1c28a7-1103-4ecc-bb3f-2391ff561ff7" />

<img width="378" height="394" alt="image" src="https://github.com/user-attachments/assets/2d6ed339-8945-4274-a958-2bcdc0fa1f69" />


Vit_Base_Small_16

<img width="569" height="226" alt="image" src="https://github.com/user-attachments/assets/cddb02dd-5221-41f7-a9d0-edfbe68cd863" />

<img width="742" height="687" alt="image" src="https://github.com/user-attachments/assets/b0c74a90-d03e-4f12-9a9f-641bc087f434" />

<img width="609" height="499" alt="image" src="https://github.com/user-attachments/assets/27e5bbc7-54c8-4ef2-a2ec-fc1f7da47b16" />

<img width="444" height="447" alt="image" src="https://github.com/user-attachments/assets/9bf93472-1b22-489e-85fa-15bd5f95b84e" />


<img width="477" height="429" alt="image" src="https://github.com/user-attachments/assets/656ec0ed-73df-4cdd-9a66-a66b6c8f9563" />


BEiT

<img width="551" height="207" alt="image" src="https://github.com/user-attachments/assets/e1a9f931-2121-430f-94d2-43613efc81f0" />

<img width="666" height="626" alt="image" src="https://github.com/user-attachments/assets/b89fbfc1-a307-41e6-96ff-b780f4f70354" />

<img width="536" height="458" alt="image" src="https://github.com/user-attachments/assets/81c8d9d8-35c5-4e2c-b05f-1b46a67c4f66" />

<img width="423" height="409" alt="image" src="https://github.com/user-attachments/assets/dcb77e43-f909-4f80-bf1c-13f0b4ead5df" />

<img width="447" height="409" alt="image" src="https://github.com/user-attachments/assets/5d1ef723-2f49-4ee2-a7f4-8aa0e4c68c22" />


Hybrid_ViT
-----------

BotNet

<img width="590" height="220" alt="image" src="https://github.com/user-attachments/assets/5c507834-bee2-4f35-a9da-06878582065c" />

<img width="732" height="678" alt="image" src="https://github.com/user-attachments/assets/e16bf234-84c8-4a69-8ac8-7e95aea795c9" />

<img width="625" height="479" alt="image" src="https://github.com/user-attachments/assets/2328cfcb-b75a-4366-8d58-e3a7e0673028" />

<img width="439" height="411" alt="image" src="https://github.com/user-attachments/assets/99aeb3dd-87ae-4ad6-a687-197872a33e59" />

<img width="486" height="415" alt="image" src="https://github.com/user-attachments/assets/98f85f7b-368f-4571-b3dc-41989472b584" />


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

swim+EfficientNetB4+inceptionV3(Ensemble)
-----------------------------------------

<img width="873" height="253" alt="image" src="https://github.com/user-attachments/assets/07df06c1-b305-4dae-b3a0-8fb89c936765" />

<img width="824" height="500" alt="image" src="https://github.com/user-attachments/assets/a6149245-0779-4c7f-965f-a6dd7d5e4f3b" />


Proposed_Model
+++++++++++++++

MSW-EnsNet


<img width="576" height="230" alt="image" src="https://github.com/user-attachments/assets/2a05c07d-505a-4c5f-af7b-152c0d8d75e0" />


<img width="392" height="420" alt="image" src="https://github.com/user-attachments/assets/45ba1e38-20aa-4133-a2c7-af6ca7d22d46" />


<img width="533" height="455" alt="image" src="https://github.com/user-attachments/assets/90772485-e9d5-4dc7-a11a-479621d053db" />





<img width="719" height="611" alt="image" src="https://github.com/user-attachments/assets/f868dd86-f83b-43b2-9eb4-53bcc360902e" />



<img width="696" height="551" alt="image" src="https://github.com/user-attachments/assets/f38152f0-1002-4284-bd03-031d001e1425" />



Analysis
++++++++


<img width="845" height="652" alt="image" src="https://github.com/user-attachments/assets/19dbbcae-c6e3-41e0-a977-e1347ed62890" />

Ablation Study 
---------------

Boosting

<img width="892" height="588" alt="image" src="https://github.com/user-attachments/assets/4241fa88-f9d0-4a7f-bfd2-dadcf3de059a" />


Weighted_Decision_Fusion 
------------------------

<img width="911" height="525" alt="image" src="https://github.com/user-attachments/assets/d5d9e38b-5d34-4b46-81d1-9f68eaa21883" />


The ablation results indicate that neither EfficientNet-B4 nor InceptionV3 alone achieves optimal performance (F1: 0.9483 and 0.9372, respectively), whereas their weighted fusion significantly improves results, peaking at F1 = 0.9551 for weights (0.7, 0.3). This confirms that integrating global and local feature representations enhances classification robustness.



w1, w2 = 0.7, 0.3
boost = torch.tensor([1.0, 1.0, 1.1, 1.1], device=device)


<img width="877" height="47" alt="image" src="https://github.com/user-attachments/assets/94bf3cf6-d28f-4378-a699-8eb7d632e2ff" />



<img width="821" height="622" alt="image" src="https://github.com/user-attachments/assets/faf6bb1c-6e14-4cf6-b995-65dd32e972f0" />





Architecture of proposed model
-----------------------------------



                        ┌────────────────────────────┐
                        │     Chest X-ray Input      │
                        │   (Single image → dual)    │
                        └────────────┬───────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │                                       │

        ┌───────────────────────┐           ┌───────────────────────┐
        │   EfficientNet-B4     │           │     Inception-V3      │
        │    (Global Branch)    │           │     (Local Branch)    │
        └──────────┬────────────┘           └──────────┬────────────┘
                   │                                   │

        ┌───────────────────────┐           ┌───────────────────────┐
        │     SE Attention      │           │    Parallel Filters   │
        │ (Squeeze & Excite)    │           │   (1x1, 3x3, 5x5)     │
        └──────────┬────────────┘           └──────────┬────────────┘
                   │                                   │

        ┌───────────────────────┐           ┌───────────────────────┐
        │    Global Features    │           │     Local Features    │
        │  - Lung symmetry      │           │  - Texture patterns   │
        │  - Diffuse opacities  │           │  - Edge irregularity  │
        └──────────┬────────────┘           └──────────┬────────────┘
                   │                                   │

        ┌───────────────────────┐           ┌───────────────────────┐
        │        Softmax        │           │        Softmax        │
        │        (P1)           │           │        (P2)           │
        └──────────┬────────────┘           └──────────┬────────────┘
                   │                                   │
                   └──────────────┬────────────────────┘
                                  │

                  ┌───────────────────────────────────┐
                  │   Weighted Decision Fusion        │
                  │   Output = 0.55 * P1 + 0.45 * P2  │
                  └────────────────┬──────────────────┘
                                   │

                  ┌───────────────────────────────────┐
                  │      Class-Aware Weighting        │
                  │  weights = [1.0, 1.0, 1.2, 1.2]   │
                  │  (Boost minority classes)         │
                  └────────────────┬──────────────────┘
                                   │

                  ┌───────────────────────────────────┐
                  │  Test-Time Augmentation (TTA)     │
                  │  (original + flips → average)     │
                  └────────────────┬──────────────────┘
                                   │

                  ┌───────────────────────────────────┐
                  │        Final Prediction           │
                  │  Normal | COVID | Bacterial | Viral│
                  └───────────────────────────────────┘









