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
        
Comparative study of CNN,VIT,Hybrid_VIT
----------------------------------------------------
*****  Training without CLAHE (Balanced class weights)  *****
---------------------------------------------------------

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


------------------------------------------------------------------------------------------------------------------------------------------------

Training with CLAHE dataset
-----------------------------


EfficientNetB4:-
------------------
<img width="665" height="170" alt="image" src="https://github.com/user-attachments/assets/22c73a34-1e77-48ca-a534-4c995b27d0ac" />

<img width="538" height="215" alt="image" src="https://github.com/user-attachments/assets/cc92e5be-bd53-4798-9301-fed5e01202d8" />

<img width="696" height="560" alt="image" src="https://github.com/user-attachments/assets/334abf06-cd11-4b2a-9a4b-fdcdc4b45380" />


<img width="583" height="440" alt="image" src="https://github.com/user-attachments/assets/197ff2a3-0d3a-4e82-940b-f706cf099169" />

<img width="419" height="405" alt="image" src="https://github.com/user-attachments/assets/93929130-e381-4c81-a8e6-2e4b5d3acf88" />

<img width="478" height="388" alt="image" src="https://github.com/user-attachments/assets/3b489832-4d63-4071-9478-c557c26ad83f" />


InceptionV3:-
---------------
<img width="563" height="279" alt="image" src="https://github.com/user-attachments/assets/b369db0a-f45e-48b3-b462-ab29b9cc0e08" />

<img width="292" height="315" alt="image" src="https://github.com/user-attachments/assets/423f6c25-4e43-4a94-bc9d-d22716b8fc5e" />

<img width="743" height="686" alt="image" src="https://github.com/user-attachments/assets/6f5867c3-04ed-4459-997f-8537eae1a396" />

<img width="645" height="483" alt="image" src="https://github.com/user-attachments/assets/f92199b6-d697-4dc9-81d1-43265fd8c8e4" />

<img width="456" height="428" alt="image" src="https://github.com/user-attachments/assets/8612a9db-c137-4ab3-8af7-a099387c67e3" />

<img width="460" height="434" alt="image" src="https://github.com/user-attachments/assets/c4234990-acad-4e9b-9ce2-4b2c0068c31e" />


MSW-EnsNet:-
------------
<img width="366" height="699" alt="image" src="https://github.com/user-attachments/assets/218b27d2-a501-4d3e-9ce9-3e3dba97b9c8" />

<img width="373" height="696" alt="image" src="https://github.com/user-attachments/assets/ebbcc7a0-18e3-4b49-a8e4-328838c9333f" />


<img width="582" height="294" alt="image" src="https://github.com/user-attachments/assets/47c7b4b5-6573-4827-b2a4-b5f54e6e170c" />

<img width="593" height="484" alt="image" src="https://github.com/user-attachments/assets/7b69a539-1c96-4228-a92e-1d66a42adebe" />

<img width="636" height="595" alt="image" src="https://github.com/user-attachments/assets/0698b363-a4e5-431a-b849-c75aeee36988" />

<img width="625" height="683" alt="image" src="https://github.com/user-attachments/assets/b6fad742-1af4-4abf-ad6b-6649c9683f05" />

<img width="404" height="326" alt="image" src="https://github.com/user-attachments/assets/b0b2e132-017e-490a-912b-80864cf790c0" />


<img width="516" height="322" alt="image" src="https://github.com/user-attachments/assets/143f255c-df03-447d-9371-01f8c6661a7f" />

<img width="541" height="287" alt="image" src="https://github.com/user-attachments/assets/a33e0599-c775-4237-8a35-1691c5d69548" />

<img width="562" height="383" alt="image" src="https://github.com/user-attachments/assets/c5ad33f8-f1b5-4ec4-acdd-ce2b4352de49" />

<img width="318" height="281" alt="image" src="https://github.com/user-attachments/assets/6fb8eacd-f8c5-4bac-8002-74e529a0814e" />

<img width="537" height="448" alt="image" src="https://github.com/user-attachments/assets/7a4bc969-dfab-459f-96f4-ad665e18d8b0" />



TransUNet

<img width="520" height="194" alt="image" src="https://github.com/user-attachments/assets/c278862e-94f9-49f6-87d7-8199d4f72008" />

<img width="629" height="612" alt="image" src="https://github.com/user-attachments/assets/204da2c1-cff0-4a47-8918-9154b02c9bff" />


<img width="632" height="242" alt="image" src="https://github.com/user-attachments/assets/63b23aee-1351-4ab3-9685-8a0473cf13ed" />

<img width="614" height="700" alt="image" src="https://github.com/user-attachments/assets/73f502a1-83b7-4c21-9663-6cc49d3c42db" />



ResViT

<img width="313" height="150" alt="image" src="https://github.com/user-attachments/assets/56e9c5cf-05dd-4977-80ea-34736a35efe5" />

<img width="579" height="463" alt="image" src="https://github.com/user-attachments/assets/295f4731-6fda-4de4-addb-5232b9732e59" />

<img width="462" height="428" alt="image" src="https://github.com/user-attachments/assets/1e3a75a7-64c2-44dd-82b8-bbca2b917c2b" />

<img width="252" height="104" alt="image" src="https://github.com/user-attachments/assets/2f59f8b5-de20-4620-b896-70868fe9685a" />




Swim Transformer


<img width="519" height="193" alt="image" src="https://github.com/user-attachments/assets/a0ecc640-e534-4ade-8d44-34db77f3183b" />

<img width="646" height="620" alt="image" src="https://github.com/user-attachments/assets/4cc1f166-4829-414d-a6e7-a24f935ed51a" />

<img width="559" height="424" alt="image" src="https://github.com/user-attachments/assets/fc08a721-5010-4f6c-a7ea-38e9287bb5ec" />

<img width="476" height="414" alt="image" src="https://github.com/user-attachments/assets/e4e69112-16ac-4970-91df-9b4a309efd03" />

<img width="329" height="175" alt="image" src="https://github.com/user-attachments/assets/feaa4b0e-c2a7-4ff7-a652-a86f9182d1a2" />


Ensemble(swim+eff+incep)


<img width="617" height="248" alt="image" src="https://github.com/user-attachments/assets/af87b19b-f4d8-4315-8035-abb90733d133" />

<img width="735" height="504" alt="image" src="https://github.com/user-attachments/assets/83c1a49a-c316-4a99-8c8b-e6f1959cf211" />

<img width="412" height="268" alt="image" src="https://github.com/user-attachments/assets/fa081c22-e3dd-4134-9223-976dd7c619d1" />




Computational Complexity
-------------------------

InceptionV3

{'Params(M)': 23.83, 'FLOPs(G)': 2.86, 'Size(MB)': 91.27, 'Inf Time(ms)': 11.03, 'FPS': 90.7, 'GPU Mem(GB)': 0.11}




EfficientNetB4

{'Params(M)': 19.34, 'FLOPs(G)': 1.58, 'Size(MB)': 74.65, 'Inf Time(ms)': 15.24, 'GPU Mem(GB)': 0.59}




Swim Transformer


{'Params(M)': 28.29, 'FLOPs(G)': 2.98, 'Size(MB)': 108.22, 'Inf Time(ms)': 12.58, 'FPS': 79.48, 'GPU Mem(GB)': 0.22}


TransUNet


{'Params(M)': 60.86, 'FLOPs(G)': 8.9, 'Size(MB)': 232.51, 'Inf Time(ms)': 11.12, 'GPU Mem(GB)': 0.52}



MSW-EnsNet


{'Params(M)': 41.3, 'FLOPs(G)': 7.33, 'Size(MB)': 158.8, 'Inf Time(ms)': 27.41, 'GPU Mem(GB)': 0.69}


ResViT

{'Params(M)': 110.75, 'FLOPs(G)': 15.42, 'Size(MB)': 422.87, 'Inf Time(ms)': 20.13, 'GPU Mem(GB)': 0.54}


External Validation 
--------------------

kermany dataset

<img width="851" height="78" alt="image" src="https://github.com/user-attachments/assets/f7c87c0b-e38f-4a48-92e0-c4ae49bc3671" />

<img width="819" height="99" alt="image" src="https://github.com/user-attachments/assets/3bcf3b51-7bc8-4928-a18d-25314f1aa333" />

covid19 dataset

<img width="831" height="136" alt="image" src="https://github.com/user-attachments/assets/6c4093ce-bee3-4de2-87bf-ced88164d7c9" />



Without Clahe on proposed model
-----------------------------------


<img width="844" height="331" alt="image" src="https://github.com/user-attachments/assets/e0e02814-1a43-40c4-9869-21e131295367" />




without label_smoothing on proposed model
-----------------------------------------






















