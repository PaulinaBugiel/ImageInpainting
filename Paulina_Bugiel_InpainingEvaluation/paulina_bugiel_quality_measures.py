# Author: Paulina Bugiel, 2024

import os
import torchvision
import torch
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader, Subset
from image_similarity_measures.evaluate import evaluation
import glob
import cv2


""" Script for image generation assesment measures. 
For general generated image quality computation use frechet_inception_distance
For computation of similarity between original and generated images use image_similarity_measures 
"""


# Compute Frechet Inception Distance between all real and generated samples
def frechet_inception_distance(images_dir, gen_label):
    trans = transforms.Compose([transforms.Resize([299,299]), transforms.ToTensor(), transforms.ConvertImageDtype(torch.uint8)])
    fid = FrechetInceptionDistance(feature=64)
    val_ds = datasets.ImageFolder(imgs_dir, transform=trans)

    real_idx = [i for i,label in enumerate(val_ds.targets) if label == val_ds.class_to_idx['real']]
    real_subset = Subset(val_ds, real_idx)
    real_loader = DataLoader(real_subset, batch_size=len(real_subset))
    for images, labels in real_loader:
        fid.update(images, real=True)

    aotgan_idx = [i for i, label in enumerate(val_ds.targets) if label == val_ds.class_to_idx[fake_label]]
    aotgan_subset = Subset(val_ds, aotgan_idx)
    aotgan_loader = DataLoader(aotgan_subset, batch_size=len(aotgan_subset))
    for images, labels in aotgan_loader:
        fid.update(images, real=False)

    frechet = fid.compute()
    print(f'FID: {frechet}')


# Compute RMSE, SSIM and FSIM average image similarity measures for all original and generated image pairs
def image_similarity_measures(images_dir, gen_label, real_label='real'):
    real_imgs_dir = os.path.join(os.path.normpath(images_dir), real_label)
    if not os.path.isdir(real_imgs_dir):
        print(f'{real_imgs_dir} \n directory does not exist')
    gen_imgs_dir = os.path.join(os.path.normpath(images_dir), gen_label)
    if not os.path.isdir(gen_imgs_dir):
        print(f'{gen_imgs_dir} \n directory does not exist')

    mean_rmse = 0.0
    mean_ssim = 0.0
    mean_fsim = 0.0
    i = 0
    for real_img_name in os.listdir(os.path.normpath(real_imgs_dir)):
        real_img_path = os.path.join(real_imgs_dir, real_img_name)
        img_basename = os.path.splitext(real_img_name)[0].split('_')[0]
        glob_string = os.path.join(gen_imgs_dir, img_basename)+'*'
        gen_file_paths = [file for file in glob.glob(glob_string)]
        if len(gen_file_paths) == 0:
            print(f'No matching generated files found for {real_img_name}. Skipping')
            continue
        elif len(gen_file_paths) > 1:
            print(f'Found more than one matching generated file. Choosing the first.')
        gen_img_path = gen_file_paths[0]
        gen_img = cv2.imread(gen_img_path)
        real_img = cv2.imread(real_img_path)
        if not gen_img.shape == real_img.shape:
            real_img = cv2.resize(real_img, gen_img.shape[:2])
            real_img_path = './tmp_real.jpg'
            cv2.imwrite(real_img_path, real_img)

        metrics = evaluation(org_img_path=real_img_path, pred_img_path=gen_img_path, metrics=['rmse', 'ssim', 'fsim'])
        mean_rmse += metrics['rmse']
        mean_ssim += metrics['ssim']
        mean_fsim += metrics['fsim']
        i = i + 1
        print(i)
    mean_rmse = mean_rmse/i
    mean_ssim = mean_ssim/i
    mean_fsim = mean_fsim/i
    print(f'MEAN RMSE: {mean_rmse}')
    print(f'MEAN SSIM: {mean_ssim}')
    print(f'MEAN FSIM: {mean_fsim}')


if __name__ == '__main__':
    imgs_dir = '../lsun_dataset/images_for_evaluation'
    fake_label = 'opencv_telea'  # class label (subdirectory name) for fake images
    print(f'Computing measures for {fake_label}')
    image_similarity_measures(imgs_dir, fake_label)
    frechet_inception_distance(imgs_dir, fake_label)
