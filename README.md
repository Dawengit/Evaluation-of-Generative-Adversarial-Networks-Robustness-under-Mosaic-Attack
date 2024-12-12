# Evaluation of Generative Adversarial Networks’ Robustness under Mosaic Attack  
**ECE661 Team 22**

## Overview
In many critical image-based applications—such as surveillance, medical imaging, and autonomous vehicles—degraded images due to blur, noise, or obstructions can severely hinder performance and reliability. Traditional restoration methods often struggle with complex degradations, underscoring the need for more advanced techniques.

This project investigates the robustness of three prominent Generative Adversarial Network (GAN) models—**DeblurGAN**, **Pix2Pix**, and **SRGAN**—against adversarial mosaic attacks. We employ a facial image dataset and simulate challenging conditions by introducing severe blurring and applying a mosaic pattern attack to the input images. Our objective is to evaluate how well these models restore image clarity and preserve key facial attributes when confronted with deliberately perturbed inputs.

## Methodology
1. **Data Preparation:**  
   - We split the dataset into training and testing sets with an 8:2 ratio.
   - GaussianBlur is applied (radius = 20) to simulate image degradation.
   - For robust testing, a mosaic attack with severity 70 is introduced to further degrade the blurred images.
   
2. **Models Implemented:**
   - **DeblurGAN:**  
     A ResNet-based generator combined with a PatchGAN discriminator, employing adversarial and perceptual losses to achieve high-fidelity image restoration.
   
   - **Pix2Pix:**  
     A versatile image-to-image translation framework using a U-Net generator and PatchGAN discriminator. It balances adversarial loss with L1 loss for stable, realistic outputs.
   
   - **SRGAN:**  
     A GAN-based super-resolution model focusing on perceptual quality. It uses a ResNet-based generator and PatchGAN discriminator, along with a perceptual loss that emphasizes high-frequency details.
   
3. **Training:**
   - All models are trained for 200 epochs using the Adam optimizer.
   - Inputs are resized to 256×256 pixels.
   - Models are first trained on blurred datasets (no attacks), and then retrained on mosaic-attacked datasets.
   
4. **Evaluation Metrics:**
   - **Peak Signal-to-Noise Ratio (PSNR):**  
     Quantifies the fidelity of the reconstructed image at the pixel level.
   - **Structural Similarity Index Measure (SSIM):**  
     Evaluates the perceptual quality by comparing luminance, contrast, and structure between the restored and original images.

## Results and Key Findings
- **DeblurGAN** achieves the highest PSNR and SSIM on non-attacked images, indicating top-tier image restoration quality under ideal conditions.
- Under mosaic attacks, all models’ performance declines, reflected in increased losses and reduced PSNR/SSIM.
- **Pix2Pix** demonstrates the smallest relative drop in metrics, indicating superior robustness against adversarial mosaic patterns.
- **SRGAN** experiences greater instability and a more substantial performance decline under attack scenarios.

## Conclusion and Future Work
Our experiments reveal that DeblurGAN excels in delivering high-fidelity restoration when images are simply blurred, while Pix2Pix provides a more balanced solution by maintaining better resilience under adversarial mosaic attacks. Future directions include evaluating these and other GAN models against a broader range of adversarial conditions (e.g., noise, occlusions, geometric distortions), integrating defensive training strategies, and improving computational efficiency to ensure practical deployment in real-world surveillance and image restoration systems.

## Acknowledgments and References
This project is part of ECE661 Team 22. For further reading on the models, please refer to:
- [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (Ledig et al.)](https://arxiv.org/abs/1609.04802)
- [Image-to-Image Translation with Conditional Adversarial Networks (Isola et al.)](https://arxiv.org/abs/1611.07004)
- [DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better (Kupyn et al.)](https://arxiv.org/abs/1908.03826)
