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


class BaseDataset(Dataset):
    def __init__(self, args, split, debugging=False, preprocesser=None):
        self.debugging = debugging
        if debugging:
            dataset_root = "/media/data/hw/Data/sort/*.png"
            self.img_size = (256, 256)
        else:
            self.args = args
            dataset_root = args.frame_root + "*.png"
            self.img_size = args.img_size
            self.max_video_len = args.max_video_len
            assert self.max_video_len == 1
            self.fps = args.fps

        dataset_roots = glob(dataset_root, recursive=True)
        # dataset_roots.sort()

        if split == "train":
            all_images = dataset_roots[: int(len(dataset_roots) * 0.99)]
        else:
            all_images = dataset_roots[int(len(dataset_roots) * 0.99) :]

        self.all_images = all_images

        self.split = split
        self.color_aug = torchvision.transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
        )
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
                    transforms.Normalize(
                        [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
                    ),
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
                    transforms.Normalize(
                        [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
                    ),
                ]
            )
            self.ref_transform_mask = transforms.Compose(
                [  # follow CLIP transform
                    transforms.RandomResizedCrop(
                        (224, 224),
                        scale=self.img_scale,
                        ratio=self.img_ratio,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.ToTensor(),
                ]
            )
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
        return Image.fromarray(img_array * mask_array), Image.fromarray(
            img_array * (1 - mask_array)
        )  # foreground, background

    def augmentation(self, frame, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(frame)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        while 1:
            # seyeon START ####################################################################################
            # img name
            img_name = self.all_images[
                idx
            ]  # /media/data/HDTF_preprocessed/30_frame/HDTF/RD_Radio10_000/01758.jpg
            condition_img_name = img_name.replace("sort", "grsam_masked").replace("image", "source")
            name_ref_img = random.choice(self.all_images)
            while name_ref_img == img_name:
                name_ref_img = random.choice(self.all_images)

            # numpy -> PIL
            input_rgb = Image.open(img_name).convert("RGB")
            input_segmentation = Image.open(condition_img_name).convert("RGB")
            input_reference_rgb = Image.open(name_ref_img).convert("RGB")

            # augmentation & reshaping
            state = torch.get_rng_state()
            label_img = self.augmentation(input_rgb, self.transform, state)
            condition_img = self.augmentation(input_segmentation, self.cond_transform, state)
            reference_img_controlnet = self.augmentation(
                input_rgb, self.transform, state
            )  # controlnet path input
            if self.debugging:
                reference_img = self.augmentation(input_reference_rgb, self.ref_transform, state)
            else:
                if getattr(self.args, "refer_clip_preprocess", None):
                    print("### Use clip preprocess ###")
                    reference_img = self.preprocesser(input_reference_rgb).pixel_values[
                        0
                    ]  # use clip preprocess
                    reference_img = torch.from_numpy(reference_img)
                else:
                    reference_img = self.augmentation(
                        input_reference_rgb, self.ref_transform, state
                    )

            gan_reference_img = self.augmentation(input_reference_rgb, self.transform, state)
            # seyeon END ####################################################################################

            # label_imgs: GT
            # cond_imgs: pose controlnet input
            # reference_img: CLIP image encoder input
            # reference_img_controlnet: 원래 background image // 필요없긴 함...!
            outputs = {
                "img_key": img_name,
                "name_ref_img": name_ref_img,
                "label_imgs": label_img,
                "cond_imgs": condition_img,
                "reference_img": reference_img,
                "gan_reference_img": gan_reference_img,
                "reference_img_controlnet": reference_img_controlnet,
            }

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
    vis_outputs = dataset.__getitem__(cnt)
    for item in vis_outputs:
        if item == "img_key" or item == "name_ref_img":
            print(vis_outputs[item])
            continue

        save_image(vis_outputs[item], f"{save_root}{str(cnt)}_{item}.jpg")


# data_root = "/media/data/HDTF_preprocessed/30_frame/HDTF/"
# split = 'train'
# get_image_list(data_root, split)

# cnt = 0
# while 1:
#     cnt+=1
#     test(cnt)
#     if cnt == 100:
#         break
