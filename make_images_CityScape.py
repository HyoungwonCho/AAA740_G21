# --------------------------------------------------------
# DisCo - Disentangled Control for Referring Human Dance Generation in Real World
# Licensed under The Apache-2.0 license License [see LICENSE for details]
# Tan Wang (TAN317@e.ntu.edu.sg)
# Work done during internship at Microsoft
# --------------------------------------------------------

from utils.wutils_ldm import *
from agent_CityScape import Agent_LDM, WarmupLinearLR, WarmupLinearConstantLR
import os
import torch
from utils.lib import *
from utils.dist import dist_init
from dataset.tsv_dataset import make_data_sampler, make_batch_data_sampler
import imageio
import subprocess
from moviepy.editor import ImageSequenceClip, VideoFileClip, AudioFileClip

torch.multiprocessing.set_sharing_strategy("file_system")
import random
from glob import glob
from PIL import Image


def get_loader_info(args, size_batch, dataset):
    is_train = dataset.split == "train"
    if is_train:
        images_per_gpu = min(size_batch * max(1, (args.max_video_len // dataset.max_video_len)), 128)
        images_per_batch = images_per_gpu * args.world_size
        iter_per_ep = len(dataset) // images_per_batch
        if args.epochs == -1:  # try to add iters into args
            assert args.ft_iters > 0
            num_iters = args.ft_iters
            args.epochs = (num_iters * images_per_batch) // len(dataset) + 1
        else:
            num_iters = iter_per_ep * args.epochs
    else:
        images_per_gpu = size_batch * (args.max_video_len // dataset.max_video_len)
        images_per_batch = images_per_gpu * args.world_size
        iter_per_ep = None
        num_iters = None
    loader_info = (images_per_gpu, images_per_batch, iter_per_ep, num_iters)
    return loader_info


def make_data_loader(args, size_batch, dataset, start_iter=0, loader_info=None):
    is_train = dataset.split == "train"
    collate_fn = None  # dataset.collate_batch
    is_distributed = args.distributed
    if is_train:
        shuffle = True
        start_iter = start_iter
    else:
        shuffle = False
        start_iter = 0
    if loader_info is None:
        loader_info = get_loader_info(args, size_batch, dataset)
    images_per_gpu, images_per_batch, iter_per_ep, num_iters = loader_info

    if hasattr(args, "limited_samples"):
        limited_samples = args.limited_samples // args.local_size
    else:
        limited_samples = -1
    random_seed = args.seed
    sampler = make_data_sampler(dataset, shuffle, is_distributed, limited_samples=limited_samples, random_seed=random_seed)
    batch_sampler = make_batch_data_sampler(sampler, images_per_gpu, num_iters, start_iter)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_sampler=batch_sampler,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    meta_info = (images_per_batch, iter_per_ep, num_iters)
    return data_loader, meta_info


def make_video(result_frames, audio_path, ldmk_identity, fps, rand):
    output_path = "eval_visualization/"
    save_path_pasted_pred = os.path.join(output_path, f"{ldmk_identity}_{str(rand)}.mp4")
    save_path_pasted_audio_pred = os.path.join(output_path, f"AUDIO_{ldmk_identity}_{str(rand)}.mp4")
    imageio.mimwrite(save_path_pasted_pred, result_frames, fps=fps, output_params=["-vf", f"fps={fps}"])
    subprocess.call(
        f"ffmpeg -y -i {save_path_pasted_pred} -i {audio_path} -map 0:v -map 1:a -c:v copy -shortest {save_path_pasted_audio_pred}",
        shell=True,
    )
    os.remove(save_path_pasted_pred)
    print(f"//////////////////////\n{save_path_pasted_audio_pred}\n//////////////////////")


def main_worker(args, rand):
    ############################################################
    ############################################################

    cf = import_filename(args.cf)  # cf config/ref_attn_clip_combine_controlnet/HDTF_kpt.py @ scripts/finetune_sdm_HDTF.sh
    Net, inner_collect_fn = cf.Net, cf.inner_collect_fn
    dataset_cf = import_filename(args.dataset_cf)  # dataset_cf = 'dataset/dataset_seyeon.py' @ config/ref_attn_clip_combine_controlnet/HDTF_kpt.py
    BaseDataset = dataset_cf.BaseDataset
    logger.info("Building models...")
    model = Net(args)
    print(f"Args: {edict(vars(args))}")
    logger.warning("Do eval_visu...")

    # name_root = HDTF_root + ID_name
    # audio_path = name_root.replace('HDTF_preprocessed/30_frame/', '') + '.mp4'

    if getattr(args, "refer_clip_preprocess", None):
        eval_dataset = BaseDataset(args, split="val", preprocesser=model.feature_extractor)
    else:
        eval_dataset = BaseDataset(
            args,
            split="val",
        )
    eval_dataloader, eval_info = make_data_loader(args, args.local_eval_batch_size, eval_dataset)

    # check BaseDataset lenth
    print(len(eval_dataset))

    evaluator = Agent_LDM(args=args, model=model)

    evaluator.move_model_to_cuda()
    if evaluator.args.dist:
        evaluator.prepare_dist_model()

    eval_meter, eval_time = evaluator.eval_fn_visualization(
        eval_dataloader,
        inner_collect_fn=inner_collect_fn,
        use_tqdm=True,
        enc_dec_only="enc_dec_only" in args.eval_save_filename,
    )
    logger.info("[Rank %s] Valid Time: %s\n %s" % (evaluator.rank, eval_time, eval_meter.avg))

    print("\n\n\nGenerating frames is DonE!\n\n\n")

    # visualization_root = f"{args.eval_save_filename}/{ID_name}/*.jpg"
    # result_frames = glob(visualization_root)
    # result_frames.sort()
    # frames = [Image.open(img) for img in result_frames]

    # make_video(frames, audio_path, ID_name, fps, rand)

    # print("\n\n\nGenerating video is DonE!\n\n\n")


if __name__ == "__main__":
    from utils.args import sharedArgs

    parsed_args = sharedArgs.parse_args()
    rand = random.randint(0, 1000000)
    main_worker(parsed_args, rand)
