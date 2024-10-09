# Author: Paulina Bugiel, 2024

import os
import torch
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader, Subset
from image_similarity_measures.evaluate import evaluation
import glob
import cv2
import argparse

""" Script for image generation assessment measures. 
For general generated image quality computation use frechet_inception_distance
For computation of similarity between original and generated images use image_similarity_measures 
"""


# Compute Frechet Inception Distance between all real and generated samples
def frechet_inception_distance(imgs_dir, gen_label, real_label='real'):
    trans = transforms.Compose(
        [transforms.Resize([299, 299]), transforms.ToTensor(), transforms.ConvertImageDtype(torch.uint8)])
    fid = FrechetInceptionDistance(feature=64)
    val_ds = datasets.ImageFolder(imgs_dir, transform=trans)

    real_idx = [i for i, label in enumerate(val_ds.targets) if label == val_ds.class_to_idx[real_label]]
    real_subset = Subset(val_ds, real_idx)
    real_loader = DataLoader(real_subset, batch_size=len(real_subset))
    for images, labels in real_loader:
        fid.update(images, real=True)

    aotgan_idx = [i for i, label in enumerate(val_ds.targets) if label == val_ds.class_to_idx[gen_label]]
    aotgan_subset = Subset(val_ds, aotgan_idx)
    aotgan_loader = DataLoader(aotgan_subset, batch_size=len(aotgan_subset))
    for images, labels in aotgan_loader:
        fid.update(images, real=False)

    frechet = fid.compute()
    print(f'FID: {frechet}')


# Compute RMSE, SSIM and FSIM average image similarity measures for all original and generated image pairs
def image_similarity_measures(imgs_dir, gen_label, real_label='real'):
    real_imgs_dir = os.path.join(os.path.normpath(imgs_dir), real_label)
    if not os.path.isdir(real_imgs_dir):
        print(f'{real_imgs_dir} \n directory does not exist')
    gen_imgs_dir = os.path.join(os.path.normpath(imgs_dir), gen_label)
    if not os.path.isdir(gen_imgs_dir):
        print(f'{gen_imgs_dir} \n directory does not exist')

    mean_rmse = 0.0
    mean_ssim = 0.0
    mean_fsim = 0.0
    i = 0
    for real_img_name in os.listdir(os.path.normpath(real_imgs_dir)):
        real_img_path = os.path.join(real_imgs_dir, real_img_name)
        img_basename = os.path.splitext(real_img_name)[0].split('_')[0]
        glob_string = os.path.join(gen_imgs_dir, img_basename) + '*'
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
    mean_rmse = mean_rmse / i
    mean_ssim = mean_ssim / i
    mean_fsim = mean_fsim / i
    print(f'MEAN RMSE: {mean_rmse}')
    print(f'MEAN SSIM: {mean_ssim}')
    print(f'MEAN FSIM: {mean_fsim}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute quality metrics on real and generated images. Metrics include'
                                                 'Frechet Inception Distance and image similarity measures - RMSE, SSIM, FSIM.'
                                                 '\nMetrics are computed in original image - generated image pairs',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--images_dir', '-d', type=str, required=True,
                        help='directory containing real and generated images, which have to stored in subdirectories eg.:'
                             '\n-/a_folder/containing/images'
                             '\n|-real'
                             '\n  |-img_01.jpg'
                             '\n  |-img_02.jpg'
                             '\n  |- ...'
                             '\n|-opencv_telea'
                             '\n  |-img_01.jpg'
                             '\n  |-img_02.jpg'
                             '\n  |- ...'
                             '\n|-aot_gan'
                             '\n  |-img_01.jpg'
                             '\n  |-img_02.jpg'
                             '\n  |- ...')
    parser.add_argument('--real_subdir', '-r', type=str, required=False,
                        help='name of subdirectory containing real images', default='real')
    parser.add_argument('--generated_subdir', '-g', type=str, required=True,
                        help='name of subdirectory containing generated images')
    args = parser.parse_args()
    images_dir = os.path.normpath(args.images_dir.strip('\n\r '))
    real_subdir = args.real_subdir
    gen_subdir = args.generated_subdir
    print(f'Computing measures for {args.generated_subdir}')
    image_similarity_measures(images_dir, args.generated_subdir)
    frechet_inception_distance(images_dir, args.generated_subdir)
