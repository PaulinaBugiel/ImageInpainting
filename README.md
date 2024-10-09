# ImageInpainting

Repository with different solutions for performing image inpainting (automatically filling a covered part of an image), as well as evaluation methods.
This includes:
- classic algorithms available in OpenCV - NS and TELEA (https://docs.opencv.org/4.x/df/d3d/tutorial_py_inpainting.html)
- Training a GAN network using CycleGAN architecture (https://junyanz.github.io/CycleGAN/)
- Using AOT-GAN network (https://github.com/researchmm/AOT-GAN-for-Inpainting)

Also, a solution for quality metrics is included. Metrics include Frechet Inception Distance (FID) and image similarity measures such as RMSE or SSIM.

Code in the repository is written in Python, and uses PyTorch and OpenCV modules, among others. 
All code was run and tested on Ubuntu.

Author: Paulina Bugiel
Created: October, 2024
