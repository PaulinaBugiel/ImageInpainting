# Author: Paulina Bugiel, 2024

import argparse
from image_masker import ImageMasker


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Make dataset composed of resized input images and corresponding masked images,'
                    'given a list or directory of images and directory with masks')
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--images', '-i', type=str, required=True,
                               help='Images directory or a path to list file containing image paths')
    required_args.add_argument('--masks', '-m', type=str, required=True,
                               help='Mask images directory or a path to list file containing mask paths')
    required_args.add_argument('--out_dir', '-o', type=str, required=True,
                               help='Output directory to save images to')
    args = parser.parse_args()

    image_masker = ImageMasker(args.images, args.masks, args.out_dir, 256, 256, concatenate_images=True)
    image_masker.mask_and_save_all_images()

    print('Done! Saved', image_masker.img_cnt, 'images')

