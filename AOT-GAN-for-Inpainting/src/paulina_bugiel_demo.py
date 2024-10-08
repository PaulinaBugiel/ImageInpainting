# Author: Paulina Bugiel, 2024

import os
import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
import argparse
import random
from model.aotgan import InpaintGenerator

r""" Modified demo from AOT-GAN-for-inpainting
This demo takes two directories - original images directory and masks directory
Each original image is masked, and then passed, along with the mask, to aotgan generator.
The generator returns inpainted image, which is displayed to the user.
"""


class AotganInpainter():
    def __init__(self, args):
        self.model = InpaintGenerator(args)
        self.model.load_state_dict(torch.load(os.path.normpath(args.weights_path), map_location="cpu"))
        self.model.eval()
        self.save_dir = args.save_dir

    @staticmethod
    def postprocess(image):
        image = torch.clamp(image, -1.0, 1.0)
        image = (image + 1) / 2.0 * 255.0
        image = image.permute(1, 2, 0)
        image = image.cpu().numpy().astype(np.uint8)
        return image

    def inpaint_images_from_folder(self, images_dir: str, masks_dir: str):
        mask_paths = os.listdir(os.path.normpath(masks_dir))
        mask_paths = [os.path.join(masks_dir, name) for name in mask_paths]
        print('To move to the next image, press any key.\nTo stop processing, kill the script :)')
        i = 0
        for image_name in os.listdir(os.path.normpath(images_dir)):
            image_path = os.path.join(images_dir, image_name)
            mask_path = random.choice(mask_paths)
            print(f'Current image:', image_path)
            self.inpaint_one_image(image_path, mask_path)
            i = i+1
            print(i)

    def inpaint_one_image(self,image_path: str, mask_path: str):
        image = cv2.imread(os.path.normpath(image_path))  # TODO save original size, and resize predicted image at the end
        image = cv2.resize(image, (512, 512))
        image_tensor = (ToTensor()(image) * 2.0 - 1.0).unsqueeze(0)
        mask = cv2.imread(os.path.normpath(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.bitwise_not(mask)
        mask_tensor = (ToTensor()(mask)).unsqueeze(0)
        masked_tensor = (image_tensor * (1 - mask_tensor).float()) + mask_tensor
        masked_np = self.postprocess(masked_tensor[0])
        cv2.imshow("original image", image)
        cv2.waitKey(0)
        cv2.imshow("masked image", masked_np)
        cv2.waitKey(0)
        print('Inpainting in progress...')
        with torch.no_grad():
            pred_tensor = self.model(masked_tensor, mask_tensor)
        pred_np = self.postprocess(pred_tensor[0])
        print('Inpainting finished')
        cv2.imshow("inpainted image", pred_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        orig_name = os.path.splitext(os.path.basename(image_path))[0] + '_orig.jpg'
        masked_name = os.path.splitext(os.path.basename(image_path))[0] + '_mask.jpg'
        generated_name = os.path.splitext(os.path.basename(image_path))[0] + '_inpainted_aotgan.jpg'
        if len(self.save_dir) > 0:
            cv2.imwrite(os.path.join(self.save_dir, orig_name), image)
            cv2.imwrite(os.path.join(self.save_dir, masked_name), masked_np)
            cv2.imwrite(os.path.join(self.save_dir, generated_name), pred_np)
        return pred_np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inpaint images, given a masked image and a corresponding mask.")
    parser.add_argument('--images_dir', '-i', type=str, required=True, help='Directory containing original images.')
    parser.add_argument('--masks_dir', '-m', type=str, required=True, help='Directory containing image masks.')
    parser.add_argument('--weights_path', '-w', type=str, required=True, help='Pretrained weights of the generator network')
    parser.add_argument('--save_dir', '-s', type=str, required=False, default='', help='If present, resulting images will be saved here')
    # args required by generator model (TODO find a way to move these elsewhere)
    parser.add_argument("--block_num", type=int, default=8, help="number of AOT blocks")
    parser.add_argument("--rates", type=str, default="1+2+4+8", help="dilation rates used in AOT block")

    args = parser.parse_args()
    args.rates = list(map(int, list(args.rates.split("+"))))
    inpainter = AotganInpainter(args)
    inpainter.inpaint_images_from_folder(args.images_dir, args.masks_dir)
    print('Done!')
