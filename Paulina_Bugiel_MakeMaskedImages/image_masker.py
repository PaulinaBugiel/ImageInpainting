# Author: Paulina Bugiel, 2024

import os
import cv2
import random
from numpy import ndarray


""" Class producing a dataset of masked images. Mainly for training of GAN network. 
Aill imagages form images_dir are masked. Masks are chosen at random from mask_dir. 
Random transforms are applied to the masks(rotation and zoom of a random part of the image). 
Resulting images of size (target_width x target_height) are then masked and saved in out_dir.
if 'concatenate_images' is True, then an aligned dataset is produced - masked and original images
are both present in the output image (utilized by eg. Pix2Pix or CycleGAN networks in 'aligned' mode)
"""

class ImageMasker:
    def __init__(self, images_dir: str, masks_dir: str, out_dir: str, target_width: int, target_height: int, concatenate_images=False):
        self.images_list = []
        self.masks = []
        self.width = target_width
        self.height = target_height
        self.img_cnt = 0
        self.concatenate = concatenate_images
        if os.path.isdir(images_dir):
            for file in os.listdir(images_dir):
                self.images_list.append(os.path.join(images_dir, file))
        elif os.path.isfile(images_dir):
            with open(images_dir) as images_list_file:
                self.images_list = images_list_file.readlines()
        if len(self.images_list) == 0:
            print('No images to process')
        else:
            print(len(self.images_list), 'images to process.')

        if os.path.isdir(masks_dir):
            for file in os.listdir(masks_dir):
                mask_image = cv2.imread(os.path.join(masks_dir, file))
                self.masks.append(mask_image)
        elif os.path.isfile(masks_dir):
            with open(masks_dir) as masks_list:
                for file_path in masks_list:
                    mask_image = cv2.imread(os.path.normpath(file_path.strip('\n\r ')))
                    self.masks.append(mask_image)
        if len(self.masks) == 0:
            print('No masks provided')
        else:
            print(len(self.masks), 'available masks')

        out_folder_name = str(self.width) + 'x' + str(self.height)
        if self.concatenate:
            self.out_concatenated_dir = os.path.join(out_dir, out_folder_name + '_concatenated')
            os.makedirs(self.out_concatenated_dir, exist_ok=True)
        else:
            self.out_resized_dir = os.path.join(out_dir, out_folder_name)
            self.out_masked_dir = os.path.join(out_dir, out_folder_name + '_masked')
            os.makedirs(self.out_resized_dir, exist_ok=True)
            os.makedirs(self.out_masked_dir, exist_ok=True)
        print('Output images will be stored in:\n', out_dir)

    def __apply_mask_transforms(self, mask_image: ndarray, out_width: int, out_height: int):
        mask_image = self.__rotate_image(mask_image)
        mask_image = self.__change_image_aspect_ratio(mask_image, out_width/out_height)
        mask_image = self.__apply_random_zoom(mask_image)
        mask_image = cv2.resize(mask_image, (out_width, out_height), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow('Resized mask', mask_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return mask_image

    @staticmethod
    def __rotate_image(image: ndarray) -> ndarray:
        h, w, ch = image.shape
        # Rotate the image by a random angle and fill the gaps with white
        rotation_angle = random.randint(0, 360)
        rot_mtx = cv2.getRotationMatrix2D((w / 2, h / 2), rotation_angle, 1)
        image = cv2.warpAffine(image, rot_mtx, (w, h), borderValue=[255, 255, 255], flags=cv2.INTER_NEAREST)
        return image

    @staticmethod
    def __apply_random_zoom(image: ndarray) -> ndarray:
        h, w, _ = image.shape
        zoom_value = random.uniform(1.0, 2.0)
        new_w = int(w*zoom_value)
        new_h = int(h*zoom_value)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # crop the image to target size
        left_offset = random.randint(0, new_w - w)
        top_offset = random.randint(0, new_h - h)
        image = image[top_offset:top_offset+h, left_offset:left_offset+w]
        return image

    @staticmethod
    def __change_image_aspect_ratio(image: ndarray, target_aspect_ratio) -> ndarray:
        # Crop the mask to match output aspect ratio - avoids unnecessary 'squeezing' of the mask
        h, w, ch = image.shape
        mask_ar = w/h
        if mask_ar > target_aspect_ratio:  # mask is wider - crop the left and right of the mask
            target_w = int(target_aspect_ratio*h)
            left_offset = random.randint(0, w - target_w)
            image = image[:, left_offset:left_offset+target_w]
        elif target_aspect_ratio > mask_ar:  # mask is taller
            target_h = int(w/target_aspect_ratio)
            top_offset = random.randint(0, h - target_h)
            image = image[top_offset:top_offset+target_h, :]
        return image

    def get_random_mask(self, target_w: int, target_h: int):
        mask = random.choice(self.masks)
        mask = self.__apply_mask_transforms(mask, target_w, target_h)
        return mask

    def mask_and_save_all_images(self):
        for img_path in self.images_list:
            image = cv2.imread(os.path.normpath(img_path.strip('\n\r ')))
            # cv2.imshow('Input image', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            image = cv2.resize(image, (self.width, self.height))
            mask = self.get_random_mask(self.width, self.height)
            masked_image = cv2.bitwise_and(image, mask)
            # cv2.imshow('masked image', result)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            out_file_name = os.path.basename(os.path.splitext(img_path)[0] + '.jpg')
            if self.concatenate:
                concatenated = cv2.hconcat([masked_image, image])
                cv2.imwrite(os.path.join(self.out_concatenated_dir, out_file_name), concatenated)
            else:
                cv2.imwrite(os.path.join(self.out_resized_dir, out_file_name), image)
                cv2.imwrite(os.path.join(self.out_masked_dir, out_file_name), masked_image)
            self.img_cnt += 1