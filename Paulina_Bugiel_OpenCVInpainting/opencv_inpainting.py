import cv2
import argparse
import os
import random


""" Code performing inpainting on all images in images_dir. Masks are taken at random from masks_dir. Algorithm: ns/telea
https://docs.opencv.org/4.x/df/d3d/tutorial_py_inpainting.html
"""
def inpaint_images_from_folder(images_dir: str, masks_dir: str, algorithm: str):
    mask_paths = os.listdir(os.path.normpath(masks_dir))
    mask_paths = [os.path.join(masks_dir, name) for name in mask_paths]
    print('To move to the next image, press any key.\nTo stop processing, kill the script :)')
    i = 0
    for image_name in os.listdir(os.path.normpath(images_dir)):
        image_path = os.path.join(images_dir, image_name)
        mask_path = random.choice(mask_paths)
        print(f'Current image:', image_path)
        perform_inpainting(image_path, mask_path, algorithm)
        i = i + 1
        print(i)


def perform_inpainting(image_path: str, mask_path:str, algorithm: str):
    image = cv2.imread(os.path.normpath(image_path))
    mask = cv2.imread(os.path.normpath(mask_path), cv2.IMREAD_GRAYSCALE)
    dest_size = (image.shape[1], image.shape[0])
    mask = cv2.resize(mask, dest_size, interpolation=cv2.INTER_NEAREST)
    mask = cv2.bitwise_not(mask)  # inverse colors
    if algorithm == 'TELEA':
        inpainted_img = cv2.inpaint(image, mask, 8, cv2.INPAINT_TELEA)
    elif algorithm == 'NS':
        inpainted_img = cv2.inpaint(image, mask, 8, cv2.INPAINT_NS)
    else:
        print(f'Invalid algorithm: {algorithm}. Choices are: [TELEA |NS]')

    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.imshow('Inpainted image', inpainted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs and views image inpainting with open-cv built-in algorithms on given image')
    parser.add_argument('--images_dir', '-i', type=str, required=True)
    parser.add_argument('--masks_dir', '-m', type=str, required=True)
    parser.add_argument('--algorithm', '-a', type=str, default='NS', choices=['TELEA', 'NS'],
                        help='Algorithm to use. Open CV provides Telea or Navier-Stokes algorithm [TELEA |NS]')
    args = parser.parse_args()
    inpaint_images_from_folder(args.images_dir, args.masks_dir, args.algorithm)
