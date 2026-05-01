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
        

Training without CLAHE (Balanced weights)
------------------------------------------

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

<img width="511" height="221" alt="image" src="https://github.com/user-attachments/assets/adc6cf84-c8a7-43af-9855-0ad421a40cee" />

<img width="635" height="554" alt="image" src="https://github.com/user-attachments/assets/e25ff913-4fd1-4a1e-99a0-def73718c379" />


<img width="578" height="430" alt="image" src="https://github.com/user-attachments/assets/6c96c218-d28c-4cd2-90f8-5a2408682e2b" />


<img width="400" height="396" alt="image" src="https://github.com/user-attachments/assets/5a0b875b-caf0-4190-bff6-f8ee8a7d9e28" />

<img width="478" height="400" alt="image" src="https://github.com/user-attachments/assets/aecb900a-9a36-4d0c-88e5-295f652c81fe" />


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

<img width="522" height="197" alt="image" src="https://github.com/user-attachments/assets/db9fbd85-36b2-4256-b1ed-4f85bf1ccffe" />


<img width="672" height="634" alt="image" src="https://github.com/user-attachments/assets/25439b81-14e2-498a-9b45-0cc72e45a0a8" />


<img width="544" height="427" alt="image" src="https://github.com/user-attachments/assets/79d97a09-169a-4082-a84f-697e30456a16" />


ConViT_Base

<img width="587" height="230" alt="image" src="https://github.com/user-attachments/assets/f6c07c5c-c158-4e77-a5b6-301317e51dbb" />

<img width="739" height="689" alt="image" src="https://github.com/user-attachments/assets/7518e058-f11f-4791-8503-5187d8f0c54c" />

<img width="601" height="507" alt="image" src="https://github.com/user-attachments/assets/41141b90-e168-4636-be36-27cbea2d58db" />

<img width="498" height="423" alt="image" src="https://github.com/user-attachments/assets/2e3ffb1d-6e45-4fad-bc31-9e28a2fe943b" />



TransUNet

<img width="537" height="212" alt="image" src="https://github.com/user-attachments/assets/8d33c9dc-16f4-4b96-9486-32a67cb51436" />

<img width="652" height="603" alt="image" src="https://github.com/user-attachments/assets/b336518a-8a87-44b0-a872-64d0b054bfa3" />

<img width="583" height="462" alt="image" src="https://github.com/user-attachments/assets/37dd7412-9b37-4afc-93c4-c4c64c40202a" />

<img width="422" height="405" alt="image" src="https://github.com/user-attachments/assets/e52e2f00-9da2-4f95-b324-fe1aa7c90e2f" />

<img width="485" height="447" alt="image" src="https://github.com/user-attachments/assets/83b562e5-0fe8-4b45-9b93-c00dcf4440d6" />



Hybrid_ResViT

<img width="486" height="347" alt="image" src="https://github.com/user-attachments/assets/4dd1e8c7-c024-4d0f-bc56-16580def5ed2" />

<img width="528" height="431" alt="image" src="https://github.com/user-attachments/assets/ed37deaf-7638-4b62-a1ad-94524b584653" />


<img width="399" height="398" alt="image" src="https://github.com/user-attachments/assets/9bea8683-27dd-46ec-b531-379b1dc6e2e6" />



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



<img width="1007" height="567" alt="image" src="https://github.com/user-attachments/assets/6b98c811-0325-4dfc-aa3a-4f53608cd3e3" />



------------------------------------------------------------------------------------------------------------------------------------------------


<img width="425" height="593" alt="image" src="https://github.com/user-attachments/assets/fd90e21b-e8f1-485f-978e-abbb5c5629e4" />


<img width="436" height="535" alt="image" src="https://github.com/user-attachments/assets/6c87db51-4782-4b85-98b1-7706fa42bd73" />



<img width="548" height="210" alt="image" src="https://github.com/user-attachments/assets/805a78b5-9a32-4691-bdf2-42f15f074f67" />


<img width="281" height="683" alt="image" src="https://github.com/user-attachments/assets/70d597a8-166b-4ca3-997a-0c8cc028d2f7" />


EffNet
Accuracy: 0.9414
Precision: 0.9426
Recall: 0.9414
F1: 0.9414
Balanced_Accuracy: 0.9414
Specificity: 0.9805
AUC: 0.9885
Kappa: 0.9218

Inception
Accuracy: 0.9086
Precision: 0.9083
Recall: 0.9086
F1: 0.9084
Balanced_Accuracy: 0.9086
Specificity: 0.9695
AUC: 0.9856
Kappa: 0.8782

Ensemble_NoBoost
Accuracy: 0.9420
Precision: 0.9433
Recall: 0.9420
F1: 0.9420
Balanced_Accuracy: 0.9420
Specificity: 0.9807
AUC: 0.9928
Kappa: 0.9226

MSW_NoTTA
Accuracy: 0.9420
Precision: 0.9433
Recall: 0.9420
F1: 0.9420
Balanced_Accuracy: 0.9420
Specificity: 0.9807
AUC: 0.9925
Kappa: 0.9226

MSW_TTA
Accuracy: 0.9463
Precision: 0.9481
Recall: 0.9463
F1: 0.9464
Balanced_Accuracy: 0.9463
Specificity: 0.9821
AUC: 0.9905
Kappa: 0.9284

---------------------------------------------------------------------------------------------------------------------------

EffNet
Accuracy: 0.9414
Precision: 0.9426
Recall: 0.9414
F1: 0.9414
Balanced_Accuracy: 0.9414
Specificity: 0.9805
AUC: 0.9885
Kappa: 0.9218

Inception
Accuracy: 0.9086
Precision: 0.9083
Recall: 0.9086
F1: 0.9084
Balanced_Accuracy: 0.9086
Specificity: 0.9695
AUC: 0.9856
Kappa: 0.8782

Ensemble_NoBoost
Accuracy: 0.9420
Precision: 0.9433
Recall: 0.9420
F1: 0.9420
Balanced_Accuracy: 0.9420
Specificity: 0.9807
AUC: 0.9928
Kappa: 0.9226

MSW_NoTTA
Accuracy: 0.9420
Precision: 0.9433
Recall: 0.9420
F1: 0.9420
Balanced_Accuracy: 0.9420
Specificity: 0.9807
AUC: 0.9925
Kappa: 0.9226

MSW_TTA
Accuracy: 0.9463
Precision: 0.9481
Recall: 0.9463
F1: 0.9464
Balanced_Accuracy: 0.9463
Specificity: 0.9821
AUC: 0.9905
Kappa: 0.9284


---------------------------------------------------------------------------------------------------------------------


<img width="516" height="124" alt="image" src="https://github.com/user-attachments/assets/b9b0a7e6-9ace-49b3-b185-2b5a6c15bdcc" />

<img width="583" height="202" alt="image" src="https://github.com/user-attachments/assets/909ddbe7-096b-4ae9-a26b-c099a7b595fd" />



Training with CLAHE dataset
-----------------------------

EfficientNetB4(with classweights):-

<img width="786" height="601" alt="image" src="https://github.com/user-attachments/assets/f355e05f-7193-4da9-8080-5e79e8ac5f5e" />

<img width="398" height="152" alt="image" src="https://github.com/user-attachments/assets/194c8a00-97fe-4d83-b56b-0924a45264da" />

<img width="627" height="458" alt="image" src="https://github.com/user-attachments/assets/d85b3f67-15f7-41f2-9276-2e346410e29e" />

<img width="810" height="274" alt="image" src="https://github.com/user-attachments/assets/cf9c9cd8-ddc6-4095-ab90-8d9fca30e828" />




EfficientNetB4(balanced class weights):-

<img width="690" height="592" alt="image" src="https://github.com/user-attachments/assets/72215d66-b761-4e5d-9600-c7638c726fc5" />


<img width="561" height="227" alt="image" src="https://github.com/user-attachments/assets/c0409802-8022-4889-abf4-e12649a79f61" />



<img width="713" height="626" alt="image" src="https://github.com/user-attachments/assets/2e3e27a8-7580-4926-95df-f89b692c96bd" />


<img width="911" height="337" alt="image" src="https://github.com/user-attachments/assets/962f8225-0559-4ffa-ac16-1d6973634c29" />

<img width="817" height="249" alt="image" src="https://github.com/user-attachments/assets/a3549c06-9e3b-431e-8834-fe3d6f1f4f27" />



InceptionV3(balanced class weights):-


<img width="419" height="141" alt="image" src="https://github.com/user-attachments/assets/635a94bf-a7a8-4c04-a9d0-9c0f42e7cc7b" />

<img width="370" height="199" alt="image" src="https://github.com/user-attachments/assets/8641be0b-267a-4f60-b557-c72b88decd17" />



<img width="620" height="219" alt="image" src="https://github.com/user-attachments/assets/9b9563f0-eb1c-4841-99e4-455ae2872174" />



<img width="603" height="453" alt="image" src="https://github.com/user-attachments/assets/6a682746-0c84-4fec-a582-945c653018a9" />


<img width="591" height="449" alt="image" src="https://github.com/user-attachments/assets/0e39da54-6454-4089-912c-e51642d44a7a" />


<img width="738" height="609" alt="image" src="https://github.com/user-attachments/assets/65288f05-228b-49eb-a452-30d7198de099" />


MSW-EnsNet(with clahe)
----------------------













