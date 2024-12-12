# Evaluation of Generative Adversarial Networks’ Robustness under Mosaic Attack

**Course:** ECE661  
**Team:** 22

## Overview

This project explores the robustness of three state-of-the-art Generative Adversarial Network (GAN) models—DeblurGAN, Pix2Pix, and SRGAN—when faced with mosaic pattern attacks. The overarching goal is to understand how adversarial obstructions, such as artificially induced mosaics, affect the performance of GAN-based image restoration models in tasks like facial image enhancement.

## Methodology

1. **Dataset Preparation:**  
   The project uses a face dataset. We first apply Gaussian blur as a baseline degradation method to simulate real-world image quality issues commonly encountered in surveillance and other visual analysis applications.

2. **Models Evaluated:**
   - **DeblurGAN:** Employs a ResNet-based generator and PatchGAN discriminator for motion-deblurring tasks, focusing on sharper and more detailed output images.
   - **Pix2Pix:** Utilizes a conditional GAN framework with a U-Net generator and PatchGAN discriminator to translate degraded images into more refined outputs.
   - **SRGAN:** Designed for super-resolution tasks, featuring a ResNet-based generator that aims to enhance the spatial resolution and reconstruct high-frequency details.

3. **Mosaic Attack Introduction:**
   After training each model on blurred images, we introduce a mosaic pattern attack to further degrade the input images. This simulates adversarial conditions that hinder the models’ ability to restore images.

4. **Training and Evaluation Setup:**
   Each model is trained for 200 epochs using the Adam optimizer. We employ common image enhancement metrics such as Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) to quantitatively assess model performance under both normal and attacked scenarios.

## Repository Structure

- `code/`  
  Contains the training scripts, model architectures, and evaluation tools.
  
- `data/`  
  Placeholder for dataset files or instructions on how to access them.
  
- `models/`  
  Saved model weights and configuration files.
  
- `results/`  
  Outputs from the experiments, including metrics and logs.

- `report/`  
  Contains the project’s report detailing the methodology, related works, experimental setup, and references.

## Getting Started

1. **Dependencies:**  
   - Python 3.x
   - PyTorch
   - NumPy
   - OpenCV
   - Other dependencies as listed in `requirements.txt` (if provided)

2. **Running the Code:**  
   Instructions for data preprocessing, training, and evaluation will be found in `code/README.md`.

## Acknowledgments

This work is a result of the ECE661 team’s efforts at Duke University. We acknowledge the support of our instructors and the availability of open-source models and frameworks that facilitated our experiments.
