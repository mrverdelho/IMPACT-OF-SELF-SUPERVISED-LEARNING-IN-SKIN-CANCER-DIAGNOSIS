# IMPACT-OF-SELF-SUPERVISED-LEARNING-IN-SKIN-CANCER-DIAGNOSIS


# About the Paper
Deep neural networks (DNNs) are the standard approach for image classification. However, they require a large amount of data and corresponding annotations. Collecting
medical data is a difficult task, due to privacy restrictions. Moreover, it is even harder to obtain the clinical labels, since these must be provided by specialists. Self-supervised learning
(SSL) has emerged as a possibility to overcome this issue, since it uses non-annotated data to pre-train the DNN. Recently SSL has been applied in the context of skin cancer.
However, the results were not conclusive. Moreover, a proper analysis of the impact of different SSL approaches is still missing. In this paper we investigate two SSL approaches:
Rotation and SimCLR. Our results highlight the benefits of applying self-supervised learning to the classification of dermoscopy images. Additionally, we demonstrate that these
approaches learn different and complementary features.

This work was published in the 2022 IEEE International Symposium on Biomedical Imaging (ISBI) - https://biomedicalimaging.org/2022/.


# Implementation
This implementation uses pieces of tf.keras and TensorFlow's core APIs. 

Here you can visualize the overview of the evaluated pipelines:

<img width="883" alt="image" src="https://user-images.githubusercontent.com/57528206/161036383-18a0329e-4373-4e35-9eff-556c50cf099d.png">

The code is divided into three main folders: 

1. 'SSL pre-training' contains the two implemented SSL pretext tasks:
- **Rotation** (https://arxiv.org/abs/1803.07728) 
- **SimCLR** (https://arxiv.org/abs/2002.05709). Note that the SimCLR pretext task code, 'Simclr_task.ipynb', was adapted from https://github.com/sayakpaul/SimCLR-in-TensorFlow-2.

2. 'Classification task' contains the different used classification models architectures: 
- **supervised baseline** - 'Classification_Baseline.ipynb'; 
- two models **initialized with the SSL pretrained weights** - 'Classification_Rotation_task.ipynb' & 'Classification_Simclr_task.ipynb';
- two models that **fused both pretext tasks** - 'Classification_Early_fusion.ipynb' & 'Classification_Late_Fusion.ipynb'. Both techniques differ at the level of fusion, early fusion concatenates the models in a feature level, while late fusion fuses the models in the classification scores levels

3. 'Visualization' contains the source code to visualize the **Grad-CAM output** for the pretained models (the SimCLR - 'Grad_CAM_SimCLR_model.ipynb' and the Rotation - 'Grad_CAM_Rotation_model.ipynb' - technique).


# Dataset
This work was implemented using the ISIC Challenge 2019 dataset. This dataset comprises 25,331 dermoscopy images, divided into 8 lesions classes: Actinic keratosis (AKIEC),
Basal cell carcinoma (BCC), Benign keratosis (BKL), Dermatofibroma (DF), Melanoma (MEL), Nevus (NV), Squamous cell carcinoma (SCC) and Vascular (VASC). These labels are only used to train the classification models.

The images were collected at different medical centers (each center generated images with different sizes, color and aspect ratio). Therefore, it was necessary to pre-process
them. This process compensated the color and allowed all the images to have the same size, while maintaining their aspect ratio: 

<img width="560" alt="image" src="https://user-images.githubusercontent.com/57528206/161036027-eac18817-41e3-4827-953d-99c5c15df705.png">

After having resized all the images to the desired size (224x224), we applied the color constancy algorithm Shades of Gray (https://www.kaggle.com/code/apacheco/shades-of-gray-color-constancy/notebook): 

<img width="559" alt="image" src="https://user-images.githubusercontent.com/57528206/161036113-e55818ae-30a6-4edd-895b-0a960f70215f.png">

In order to compare the different initialization approaches and assess their robustness, we adopted a 5-time Monte Carlo sampling strategy, where the ISIC 2019 dataset was partitioned five times into training (70%) and validation (30%) sets.  
