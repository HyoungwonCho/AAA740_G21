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
import pdb
import einops
import cv2
from tqdm import tqdm
import sys
from os import path
import json

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import cv2
import numpy as np
import torch
from einops import rearrange


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)


class BaseDataset(Dataset):
    def __init__(self, args, split, debugging=False, preprocesser=None):
        self.debugging = debugging

        self.apply_canny = CannyDetector()
        # self.apply_midas = MidasDetector()

        if debugging:
            self.frame_root = os.path.join("/media/data/bdd100k/images/100k/", split)
            dataset_root = os.path.join(self.frame_root, "*.jpg")
            self.img_size = (256, 256)
        else:
            self.args = args
            self.frame_root = os.path.join(args.frame_root, split)
            dataset_root = os.path.join(self.frame_root, "*.jpg")
            self.img_size = args.img_size
            self.max_video_len = args.max_video_len
            assert self.max_video_len == 1
            self.fps = args.fps

        # load json file
        if split == "train":
            with open("/media/data/bdd100k/labels/bdd100k_labels_images_train.json") as json_file:
                labels = json.load(json_file)
                classes_name = ["daytime", "night"]
                labels = [label for label in labels if label["attributes"]["timeofday"] in classes_name]
        else:
            with open("/media/data/bdd100k/labels/bdd100k_labels_images_val.json") as json_file:
                labels = json.load(json_file)
                classes_name = ["daytime", "night"]
                labels = [label for label in labels if label["attributes"]["timeofday"] in classes_name]

            # (Pdb) json_data[0].keys() = dict_keys(['name', 'attributes', 'timestamp', 'labels'])

        dataset_roots = [img for img in glob(dataset_root, recursive=True) if os.path.isfile(img)]
        dataset_roots.sort()
        random.seed(90)
        random.shuffle(dataset_roots)

        self.all_images = dataset_roots
        self.labels = labels
        self.split = split
        self.color_aug = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        self.img_ratio = (1.0, 1.0)
        self.img_scale = (1.0, 1.0)
        self.is_composite = False
        self.size_frame = 1

        print("\n======================len_dataest======================\n", self.all_images)

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

        self.mask_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
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

    def process_mask(self, mask, img_key=None):  # pil, pil
        mask_array = np.array(mask)

        gray_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
        ret, bin_mask = cv2.threshold(gray_array, 0, 255, cv2.THRESH_BINARY)
        bin_mask[bin_mask == 255] = 1
        bin_mask = einops.repeat(bin_mask, "h w -> h w c", c=3)

        return bin_mask

    def HWC3(self, x):
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

    def augmentation(self, frame, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(frame)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        while 1:
            # Hyoungwon START ####################################################################################
            state = torch.get_rng_state()

            """load label"""
            label = self.labels[idx]
            img_name = os.path.join(self.frame_root, self.labels[idx]["name"])

            """processing label"""
            including_classes = ["bus", "car", "truck"]
            label_bbox = label["labels"]
            bbox = [label_bbox[i] for i in range(len(label_bbox)) if label_bbox[i]["category"] in including_classes]
            object_names = [box["category"] for box in bbox]
            class_name = label["attributes"]["timeofday"]

            """label image"""
            input_rgb = Image.open(img_name).convert("RGB")
            label_img = self.augmentation(input_rgb, self.transform, state)  # label_img: GT

            """cond image"""
            # cond_rgb = Image.open(cond_img_name).convert("RGB")
            # cond_img = self.augmentation(cond_rgb, self.cond_transform, state)

            """reference image"""
            random.seed()
            name_ref_img = random.choice([la["name"] for la in self.labels if la["attributes"]["timeofday"] == class_name and la["name"] != img_name])
            name_ref_img = os.path.join(self.frame_root, name_ref_img)
            input_reference_rgb = Image.open(name_ref_img).convert("RGB")
            if self.debugging:
                reference_img = self.augmentation(input_reference_rgb, self.ref_transform, state)
            else:
                if getattr(self.args, "refer_clip_preprocess", None):
                    reference_img = self.preprocesser(input_reference_rgb).pixel_values[0]
                    reference_img = torch.from_numpy(reference_img)
                else:
                    reference_img = self.augmentation(input_reference_rgb, self.ref_transform, state)  # reference_img: CLIP image encoder input

            """gan reference image"""
            gan_reference_img = self.augmentation(input_reference_rgb, self.transform, state)

            outputs = {
                "img_key": img_name,
                "name_ref_img": name_ref_img,
                "label_imgs": label_img,
                "gan_reference_img": gan_reference_img,
                # "cond_imgs": cond_img,
                "reference_img": reference_img,
            }

            return outputs


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


def reverse_channels(image):
    """Given a numpy array in RGB form, convert to BGR.  Will also convert from BGR to RGB."""
    # im[:,:,::-1] is a neat hack to convert BGR to RGB by reversing the indexing order.
    # im[:,:,::[2,1,0]] would also work but makes a copy of the data.
    return image[:, :, ::-1]


def test(cnt):
    dataset = BaseDataset(args=None, split="train", debugging=True)
    # import pdb; pdb.set_trace()
    save_root = "examples_of_dataset/"
    os.makedirs(save_root, exist_ok=True)

    vis_outputs = dataset.__getitem__(cnt)
    for item in vis_outputs:
        if item == "img_key" or item == "name_ref_img":
            print(vis_outputs[item])
            continue
        # if item == 'a' or item == 'b':
        #     vis_outputs[item].save(f'{save_root}{str(cnt)}_{item}.jpg')
        #     continue
        print(item)
        save_image(vis_outputs[item], f"{save_root}{str(cnt)}_{item}.jpg")


# cnt = 0
# while 1:
#     cnt += 1
#     test(cnt)
#     if cnt == 3:
#         break
