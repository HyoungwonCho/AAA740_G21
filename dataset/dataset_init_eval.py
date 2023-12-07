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
from ControlNet.annotator.midas.api import MiDaSInference
from ControlNet.annotator.util import resize_image


def get_image_list(data_root, num_frames):
    data_root = data_root + "*.png"
    filelist = glob(data_root)
    filelist.sort()

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


def get_INIT_image_list(data_root, wea_class, num_frames):
    data_root = data_root + f"{wea_class}/*.png"
    filelist = glob(data_root)
    # filelist.sort()

    return filelist[:num_frames]


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)


class MidasDetector:
    def __init__(self):
        self.model = MiDaSInference(model_type="dpt_hybrid").cuda()
        # self.model = MiDaSInference(model_type="dpt_hybrid")

    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1):
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().cuda()
            # image_depth = torch.from_numpy(image_depth).float()
            image_depth = image_depth / 127.5 - 1.0
            image_depth = rearrange(image_depth, "h w c -> 1 c h w")
            depth = self.model(image_depth)[0]

            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_pt = depth_pt.cpu().numpy()
            depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

            depth_np = depth.cpu().numpy()
            x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
            y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
            z = np.ones_like(x) * a
            x[depth_pt < bg_th] = 0
            y[depth_pt < bg_th] = 0
            normal = np.stack([x, y, z], axis=2)
            normal /= np.sum(normal**2.0, axis=2, keepdims=True) ** 0.5
            normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

            return depth_image, normal_image


class BaseDataset(Dataset):
    def __init__(self, args, split, debugging=False, preprocesser=None):
        self.debugging = debugging
        if debugging:
            # CityScape_data_root = "/media/data/init_hw/image/"
            # Synthesia_annot_root = "/media/data/hw/Data/syn/grsam_eval_masked/"
            self.ref_frame_root = "/media/dataset/init_hw/image/"
            self.label_frame_root = "/media/dataset/hw/Data/syn/eval/"
            self.annot_frame_root = "/media/dataset/hw/Data/syn/grsam_eval_masked/"
            weather = "sunny"
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

        all_ref_images = get_INIT_image_list(self.ref_frame_root, weather, self.num_frames)
        all_label_images = get_image_list(self.label_frame_root, self.num_frames)
        all_annot_images = get_image_list(self.annot_frame_root, self.num_frames)

        self.all_ref_images = all_ref_images
        self.all_label_images = all_label_images
        self.all_annot_images = all_annot_images

        self.split = split
        self.color_aug = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        self.img_ratio = (1.0, 1.0)
        self.img_scale = (1.0, 1.0)
        self.is_composite = False
        self.size_frame = 1

        self.apply_canny = CannyDetector()

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
        return len(self.all_ref_images)

    def __getitem__(self, idx):
        while 1:
            # seyeon START ####################################################################################
            # img name
            condition_img_name = self.all_annot_images[idx]
            label_img_name = self.all_label_images[idx]
            # depth_img_name = label_img_name.replace("/sort/", "/depth/")
            name_ref_img = random.choice(self.all_ref_images)  # /media/data/HDTF_preprocessed/30_frame/HDTF/RD_Radio10_000/01758.jpg

            # numpy -> PIL
            input_segmentation = Image.open(condition_img_name).convert("RGB")
            label = Image.open(label_img_name).convert("RGB")
            # depth_rgb = Image.open(depth_img_name).convert("RGB")
            input_reference_rgb = Image.open(name_ref_img).convert("RGB")

            """ Extract Edge & Depth """
            # label_rgb_np = np.array(label)
            # extract edges
            # detected_map_edge = self.apply_canny(label_rgb_np, 100, 200)
            # detected_map_edge = Image.fromarray(HWC3(detected_map_edge))
            # extract depth
            # detected_map_depth = depth_rgb

            # augmentation & reshaping
            state = torch.get_rng_state()
            label_img = self.augmentation(label, self.transform, state)
            condition_seg_img = self.augmentation(input_segmentation, self.cond_transform, state)
            # condition_edge_img = self.augmentation(detected_map_edge, self.cond_transform, state)
            # condition_depth_img = self.augmentation(detected_map_depth, self.cond_transform, state)

            if self.debugging:
                reference_img = self.augmentation(input_reference_rgb, self.ref_transform, state)
            else:
                if getattr(self.args, "refer_clip_preprocess", None):
                    reference_img = self.preprocesser(input_reference_rgb).pixel_values[0]  # use clip preprocess
                    reference_img = torch.from_numpy(reference_img)
                else:
                    reference_img = self.augmentation(input_reference_rgb, self.ref_transform, state)
            gan_reference_img = self.augmentation(input_reference_rgb, self.transform, state)

            # seyeon END ####################################################################################

            # label_imgs: GT
            # cond_imgs: pose controlnet input
            # reference_img: CLIP image encoder input
            # reference_img_controlnet: 원래 background image // 필요없긴 함...!

            outputs = {
                "img_key": label_img_name,
                "name_ref_img": name_ref_img,
                "label_imgs": label_img,
                "gan_reference_img": gan_reference_img,
                "cond_imgs": condition_seg_img,
                # "cond_edge_imgs": condition_edge_img,
                # "cond_depth_imgs": condition_depth_img,
                "reference_img": reference_img,
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


# data_root = "/media/dataset/hw/Data/syn/eval/"
# split = 'train'
# get_image_list(data_root, split)
