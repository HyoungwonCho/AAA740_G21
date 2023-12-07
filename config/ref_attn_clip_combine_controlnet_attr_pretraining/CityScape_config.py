import torch

from config import *

# from config.ref_attn_clip_combine_controlnet_attr_pretraining.net import Net, inner_collect_fn
from config.ref_attn_clip_combine_controlnet_attr_pretraining.net_poseControl_cont import (
    Net,
    inner_collect_fn,
)


class Args(BasicArgs):
    task_name, method_name = BasicArgs.parse_config_name(__file__)
    print(task_name)
    print(method_name)
    log_dir = os.path.join(BasicArgs.root_dir, task_name, "cont_0816")

    # data
    dataset_cf = "dataset/dataset_CityScape.py"
    max_train_samples = None
    max_eval_samples = None
    # max_eval_samples = 2
    max_video_len = 1  # L=16
    debug_max_video_len = 1
    img_full_size = (256, 256)
    img_size = (256, 256)
    fps = 5
    # WT: for tiktok image data dir
    frame_root = "/media/data/i2i_dataset/Cityscape/sort/"
    # tiktok_ann_root = 'keli/dataset/TikTok_dataset/pair_ann'
    refer_sdvae = False

    eval_before_train = False
    eval_enc_dec_only = False

    # training
    local_train_batch_size = 2
    local_eval_batch_size = 2
    learning_rate = 3e-5
    null_caption = False
    refer_sdvae = False

    # max_norm = 1.0
    epochs = 50
    num_workers = 4
    eval_step = 5
    save_step = 5
    drop_text = 1.0  # drop text only activate in args.null_caption, default=1.0
    scale_factor = 0.18215
    # pretrained_model_path = os.path.join(BasicArgs.root_dir, 'diffusers/stable-diffusion-v2-1')
    # pretrained_model_path = os.path.join(BasicArgs.root_dir, 'diffusers/sd-image-variations-diffusers')
    # sd15_path = os.path.join(BasicArgs.root_dir, 'diffusers/stable-diffusion-v1-5-2')
    pretrained_model_path = "/media/data/sd-image-variations-diffusers"
    # sd15_path = "/media/data/stable-diffusion-v1-5"
    gradient_checkpointing = True
    # enable_xformers_memory_efficient_attention = True
    freeze_unet = True

    # sample
    num_inf_images_per_prompt = 1
    num_inference_steps = 50
    guidance_scale = 7.5

    # others
    seed = 42
    # set_seed(seed)


args = Args()
