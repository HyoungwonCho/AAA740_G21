import os
import sys
from os import path

import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw
from einops import rearrange, reduce, repeat
import cv2

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from ControlNet.annotator.midas.api import MiDaSInference
from ControlNet.annotator.util import resize_image


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


def _extract_depth(input_fn, input_rgb_np):
    output_save_fn = input_fn.replace("/init/", "/init_depth/")
    # if os.path.exists(output_save_fn):
    #     return
    detected_map_depth, _ = apply_midas(resize_image(input_rgb_np, 384))
    detected_map_depth = Image.fromarray(HWC3(detected_map_depth))
    if not os.path.exists(os.path.dirname(output_save_fn)):
        os.makedirs(os.path.dirname(output_save_fn), exist_ok=True)
    print(output_save_fn)
    detected_map_depth.save(output_save_fn)


""" Extract Depth for ControlNet training """
apply_midas = MidasDetector()
frame_root = "/media/data1/init/**"
frames_root = glob(os.path.join(frame_root, "*.png"), recursive=True)
print(len(frames_root))
frames_root.sort(reverse=False)

frames_root = [
    "/media/data1/init/night/video_data_20180703/jig_KYT01/20180622b_KYT/dw_2018_06_22_21-49-30_000000_20fps_bae/camera_2_center_fov60.h264/1_00407.png"
]

for frame in tqdm(frames_root, leave=False):
    _extract_depth(frame, np.array(Image.open(frame).convert("RGB")))
