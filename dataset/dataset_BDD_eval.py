import json
import cv2
import numpy as np
import os, random, cv2, argparse
import pickle
from glob import glob
from PIL import Image, ImageDraw
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset
import torchvision.utils as vutils
import random

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from einops import rearrange

# from ControlNet.annotator.midas.api import MiDaSInference
from ControlNet.annotator.util import resize_image
import pdb


def get_image_list(data_dir, num_frames):
    filelist1 = glob(data_dir + "*.png")
    filelist2 = glob(data_dir + "*.jpg")
    filelist1.sort()
    filelist2.sort()

    filelist = filelist1 + filelist2

    random.seed(0)
    random.shuffle(filelist)
    print(filelist[:10])
    return filelist[:num_frames]


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


class BaseDataset(Dataset):
    def __init__(self, args, split, debugging=False, preprocesser=None):
        self.debugging = debugging
        if debugging:
            # CityScape_data_root = "/media/data/init_hw/image/"
            # Synthesia_annot_root = "/media/data/hw/Data/syn/grsam_eval_masked/"
            self.ref_frame_root = "/media/data/bdd10k_hw/daytime/images/"
            self.label_frame_root = "/media/data/gta5_hw_final/images/"
            self.annot_frame_root = "/media/data/gta5_hw_final/masks/"
            self.img_size = (256, 256)
            self.num_frames = 100
        else:
            self.args = args
            self.ref_frame_root = args.CityScape_data_root  # Config file에서 바꿔줬음!!! INIT root입니다.
            self.label_frame_root = args.Synthesia_root
            self.annot_frame_root = args.Synthesia_annot_root
            self.img_size = args.img_size
            self.max_video_len = args.max_video_len
            assert self.max_video_len == 1
            self.fps = args.fps
            self.num_frames = args.max_eval_samples
            weather = args.eval_save_filename.split("/")[-1]

        all_ref_images = get_image_list(self.ref_frame_root, self.num_frames)
        all_label_images = get_image_list(self.label_frame_root, self.num_frames)
        all_annot_images = get_image_list(self.annot_frame_root, self.num_frames)

        random.seed(0)
        random.shuffle(all_ref_images)
        self.all_ref_images = all_ref_images
        self.all_label_images = all_label_images
        self.all_annot_images = all_annot_images

        print(len(self.all_ref_images))
        print(len(self.all_label_images))
        print(len(self.all_annot_images))

        """ If you wanna use only the images that are not generated yet, use this code """
        # already_generated = os.listdir("/media/data/run_test/eval_visualization/BDD/daytime_without_edge/conditions/")
        # yet_label_images = [label_img for label_img in all_label_images if label_img.split("/")[-1] not in already_generated]
        # yet_annot_images = [annot_img for annot_img in all_annot_images if annot_img.split("/")[-1] not in already_generated]

        # self.all_ref_images = all_ref_images
        # self.all_label_images = yet_label_images
        # self.all_annot_images = yet_annot_images

        self.split = split
        self.color_aug = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        self.img_ratio = (1.0, 1.0)
        self.img_scale = (1.0, 1.0)
        self.is_composite = False
        self.size_frame = 1

        self.preprocesser = preprocesser
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        try:
            self.ref_transform = transforms.Compose(
                [  # follow CLIP transform
                    transforms.ToTensor(),
                    transforms.RandomResizedCrop(
                        (224, 224),
                        scale=self.img_scale,
                        ratio=self.img_ratio,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=False,
                    ),
                    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                ]
            )
            self.ref_transform_mask = transforms.Compose(
                [  # follow CLIP transform
                    transforms.RandomResizedCrop(
                        (224, 224),
                        scale=self.img_scale,
                        ratio=self.img_ratio,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=False,
                    ),
                    transforms.ToTensor(),
                ]
            )
        except:
            print("### Current pt version not support antialias, thus remove it! ###")
            self.ref_transform = transforms.Compose(
                [  # follow CLIP transform
                    transforms.ToTensor(),
                    transforms.RandomResizedCrop(
                        (224, 224),
                        scale=self.img_scale,
                        ratio=self.img_ratio,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                ]
            )
            # self.ref_transform_mask = transforms.Compose([ # follow CLIP transform
            #     transforms.RandomResizedCrop(
            #         (224, 224),
            #         scale=self.img_scale, ratio=self.img_ratio,
            #         interpolation=transforms.InterpolationMode.BICUBIC),
            #     transforms.ToTensor(),
            # ])
        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

    def add_mask_to_img(self, img, mask, img_key):  # pil, pil
        if not img.size == mask.size:
            # print(f'Reference image ({img_key}) size ({img.size}) is different from the mask size ({mask.size}), therefore try to resize the mask')
            mask = mask.resize(img.size)  # resize the mask
        mask_array = np.array(mask)
        img_array = np.array(img)
        mask_array[mask_array < 127.5] = 0
        mask_array[mask_array > 127.5] = 1
        return Image.fromarray(img_array * mask_array), Image.fromarray(img_array * (1 - mask_array))  # foreground, background

    def augmentation(self, frame, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(frame)

    def __len__(self):
        return len(self.all_annot_images)

    def __getitem__(self, idx):
        while 1:
            # seyeon START ####################################################################################
            # img name
            condition_img_name = self.all_annot_images[idx]
            label_img_name = self.all_label_images[idx]

            if idx >= len(self.all_ref_images):
                ref_idx = idx % len(self.all_ref_images)
                name_ref_img = self.all_ref_images[ref_idx]  # /media/data/HDTF_preprocessed/30_frame/HDTF/RD_Radio10_000/01758.jpg
            else:
                name_ref_img = self.all_ref_images[idx]

            # numpy -> PIL
            input_segmentation = Image.open(condition_img_name).convert("RGB")
            label = Image.open(label_img_name).convert("RGB")
            input_reference_rgb = Image.open(name_ref_img).convert("RGB")

            # augmentation & reshaping
            state = torch.get_rng_state()
            label_img = self.augmentation(label, self.transform, state)
            condition_img = self.augmentation(input_segmentation, self.cond_transform, state)

            if self.debugging:
                reference_img = self.augmentation(input_reference_rgb, self.ref_transform, state)
            else:
                if getattr(self.args, "refer_clip_preprocess", None):
                    reference_img = self.preprocesser(input_reference_rgb).pixel_values[0]  # use clip preprocess
                    reference_img = torch.from_numpy(reference_img)
                else:
                    reference_img = self.augmentation(input_reference_rgb, self.ref_transform, state)

            # seyeon END ####################################################################################

            # label_imgs: GT
            # cond_imgs: pose controlnet input
            # reference_img: CLIP image encoder input
            # reference_img_controlnet: 원래 background image // 필요없긴 함...!

            outputs = {
                "img_key": condition_img_name,
                "name_ref_img": name_ref_img,
                "label_imgs": label_img,
                # "gan_reference_img": gan_reference_img,
                "cond_imgs": condition_img,
                "reference_img": reference_img,
                # "bin_mask": bin_mask,
                # "foreground_label_img": foreground_label_img,
            }

            # outputs = {
            #     "img_key": condition_img_name,
            #     "name_ref_img": name_ref_img,
            #     "label_imgs": label_img,
            #     "cond_imgs": condition_img,
            #     "reference_img": reference_img,
            # }

            return outputs


def reverse_channels(image):
    """Given a numpy array in RGB form, convert to BGR.  Will also convert from BGR to RGB."""
    # im[:,:,::-1] is a neat hack to convert BGR to RGB by reversing the indexing order.
    # im[:,:,::[2,1,0]] would also work but makes a copy of the data.
    return image[:, :, ::-1]


def test(cnt):
    dataset = BaseDataset(args=None, split="train", debugging=True)
    # import pdb; pdb.set_trace()
    save_root = "examples_of_dataset/"
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)
    vis_outputs = dataset.__getitem__(cnt)
    for item in vis_outputs:
        if item == "img_key" or item == "name_ref_img":
            print(vis_outputs[item])
            continue

        save_image(vis_outputs[item], f"{save_root}{str(cnt)}_{item}.jpg")


# cnt = 0
# while 1:
#     cnt += 1
#     test(cnt)
#     if cnt == 100:
#         break
