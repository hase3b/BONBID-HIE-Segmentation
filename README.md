# Multimodal HIE Lesion Segmentation in Neonates: A Study of Loss Functions
This repository implements a 3D U-Net for the segmentation of Hypoxic-Ischemic Encephalopathy (HIE) lesions in neonatal MRI scans. The project utilizes the BONBID-HIE dataset and explores the optimization of loss functions for improved segmentation accuracy. Specifically, we investigate several loss functions including Dice Loss, Dice-Focal Loss, Tversky Loss, and Hausdorff Distance Loss, along with two novel hybrid loss functions that combine Dice-Focal and Tversky Loss with Hausdorff Distance Loss. Our findings highlight the superior performance of the Tversky-HausdorffDT Loss.

This is a term project for Deep Learning (96943) Fall 24' course taught by Dr. Tahir Syed at the Institute of Business Administration (IBA) Karachi. It's a collaborative effort of Abdul Haseeb and Annayah Usman.

## **Workflow**
![High Level Overview](https://github.com/user-attachments/assets/9f4e1421-1c64-40a1-a57d-1566fd4647c3)
*Figure 1: Brief Overview of the Project*

## **Visualizing The Problem**
![3D ADC Map](https://github.com/user-attachments/assets/12d1a668-e3e4-4316-9b06-2a04861f151a)
*Figure 2: Sample 3D ADC Map*
![3D ZADC Map](https://github.com/user-attachments/assets/40a21a40-2698-4514-b19c-d9bf228bed42)
*Figure 3: Sample 3D ZADC Map*
![3D Binary Label Mask](https://github.com/user-attachments/assets/0a03b60f-68cc-4479-a8e2-c8e91b214d15)
*Figure 4: Sample 3D Binary Label Mask*
![ADC_ZADC_GT](https://github.com/user-attachments/assets/f66316ed-b2b7-4f92-b061-345b5364e01a)
*Figure 5: Ground Truth Imposed on ADC and ZADC Maps Across Axial Slices*

## **Data and Preprocessing**
BONBID-HIE MICCAI 2023 Challenge dataset is used, consisting of 3D Apparent Diffusion Coefficient (ADC) maps, Z-score normalized ADC maps (ZADC), and binary label masks for 133 HIE patients.
* Preprocessing Steps:
  * Resampling the ADC and ZADC maps to a fixed size of (192, 192, 32).
  * Intensity normalization of the resampled maps using mean and standard deviation.
  * Concatenation of the ADC and ZADC maps to create a 2-channel input for the U-Net model.
* Data Augmentation:
  * Random Noise
  * Random Anisotropy
  * Random Blur
  * Random Gamma
  * Random Elastic Transformation
These augmentations simulate real-world imaging variations and improve the model's generalizability. More details on this in the "Term Paper.PDF".

## **Model Architecture**
A 3D U-Net architecture is employed due to its strong inductive bias, especially when dealing with small datasets. The model consists of:
* Three encoder and decoder blocks.
* Each encoder block includes double 3D convolutional layers, batch normalization, LeakyReLU activation, dropout, and max-pooling.
* The decoder block progressively upsamples using transpose convolutions, integrating encoder features via skip connections.
The model uses batch normalization for faster convergence and dropout to reduce overfitting. LeakyReLU ensures gradient flow through non-lesion areas, which is critical for sparse lesion segmentation.

## **Loss Functions**
We explore and compare the following loss functions:
* Dice Loss: Measures the overlap between predicted and ground truth masks.
* Dice-Focal Loss: Combines Dice Loss with Focal Loss to prioritize hard-to-classify examples.
* Tversky Loss: Generalizes Dice Loss to handle class imbalances, particularly useful for medical segmentation.
* Hausdorff Distance Loss (HDTL): Measures the distance between predicted and true boundaries for more precise segmentation.
* Hybrid Loss Functions:
  * DiceFocal-HausdorffDT Loss: Combines Dice-Focal Loss with Hausdorff Distance Loss.
  * Tversky-HausdorffDT Loss: Combines Tversky Loss with Hausdorff Distance Loss.

## **Evaluation Metrics**
Segmentation performance is evaluated using the following metrics:
* Dice Coefficient: Measures the volumetric overlap between the predicted and ground truth masks.
* Mean Surface Distance (MSD): Measures the average distance between the surfaces of predicted and true masks.
* Normalized Surface Dice (NSD): Evaluates the similarity between the boundary surfaces of predicted and ground truth masks.

## **Results**
| Loss Function               | Dice ↑ | MSD ↓   | NSD ↑  | Epochs |
|-----------------------------|--------|---------|--------|--------|
| Dice Loss (Baseline)        | 0.3800 | 15.0650 | 0.3850 | 32     |
| Dice Focal Loss             | 0.4900 | 1.7925  | 0.5275 | 49     |
| Tversky Loss                | 0.3525 | 15.3650 | 0.3375 | 38     |
| HausdorffDT Loss            | 0.3300 | Inf     | 0.2800 | 29     |
| DiceFocal-HausdorffDT Loss  | 0.4925 | 1.4225  | 0.5300 | 72     |
| Tversky-HausdorffDT Loss    | 0.5000 | 1.6250  | 0.5325 | 59     |

*Table 1: Metric-Based Comparison of Loss Functions*

![Loss Function Comparison (Visualized)](https://github.com/user-attachments/assets/8a9c5b0e-6584-429f-9818-c1086981d680)
*Figure 6: Visualizing Segmentation Masks Across Loss Functions*

* Hybrid loss functions perform better than standalone loss function but take longer to converge.
* Tversky-HausdorffDT Loss achieved the highest performance in terms of Dice and NSD while maintaining competitive MSD.

## **Limitations**
* Small lesions (< 1% volume) are difficult to segment accurately due to dataset heterogeneity.
* Evaluation was limited to the validation set, and the performance on the test set remains unknown.
* Resampling of label masks could introduce distortion in segmentation regions.

## **Future Work**
* Defining a custom loss function specifically for HIE lesions.
* Exploring the use of zero-shot models like MedSAM-2 for improved segmentation without the need for extensive training data.

To get a comprehensive understanding of the project, please refer to the "Term Paper.PDF" in the repo. Refer to "Preprocessing.ipynb" for code related to preprocessing and "Main.ipynb" for the rest of the code.

## **Repository Contents**
| Folder/File                                          | Description                                                                                                                            |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| /Corpus/ | Contains scrapped judgements in PDF & Text format along with its metadata as a CSV file. Also, contains merged content and metadata CSV file as well as final cleaned corpus CSV. |
| /Corpus Generation & Preprocessing/DocScrapper.ipynb | Scrapes SCP judgments and associated metadata from the official website.     |
| /Corpus Generation & Preprocessing/PDF2TXT&CSV.ipynb | Converts judgment PDFs to text, performs OCR, and links metadata.     |
| /Corpus Generation & Preprocessing/TextPreprocessing.ipynb | Preprocesses text to clean, normalize, and structure legal documents.     |
| /SCPRAG.ipynb | Implements the RAG pipeline, chunking, embedding, and evaluation.     |
| /Report.PDF    | Detailed report explaining the workflow, experiments, and results.    |


## **Acknowledgments**
* Tools: NumPy, Matplotlib, ImageIO, Torch, TorchIO, SimpleITK, MONAI (Metrics, Transforms, Losses), Pandas, TorchSummary
* Model(s): 3D U-Net
* Data Source: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10602767.svg)](https://doi.org/10.5281/zenodo.10602767)
* Instructor: Dr Tahir Syed (Professor IBA Karachi)
