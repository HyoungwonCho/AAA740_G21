import torch
from utils.dist import synchronize, get_rank

from config import *
from typing import Callable, List, Optional, Union

import inspect
from typing import Callable, List, Optional, Union

import torch
import random
from packaging import version
from transformers import (
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    CLIPTextModel,
)
from tqdm.auto import tqdm

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.schedulers import (
    DDIMScheduler,
    PNDMScheduler,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.utils import deprecate, PIL_INTERPOLATION
from diffusers.utils.import_utils import is_xformers_available
from torchvision import transforms, models
from torchvision.utils import save_image

# from diffusers.models import UNet2DConditionModel
from .unet_2d_condition import UNet2DConditionModel
import PIL.Image
from .controlnet import ControlNetModel, MultiControlNetModel_MultiHiddenStates

# from PIL import Image
from utils.common import ensure_directory
from utils.dist import synchronize
from loss.losses import zecon_loss_direct
from loss.metrics_accumulator import MetricsAccumulator

from .discrminiator import PatchDiscriminator  # EDIT
from .feature_extractor import FeatureExtractorDDPM

import pdb


class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        """ Load Models for DDIM inversion... Not sure for ddim_inversion_scheduler's pretrained path """
        self.ddim_inversion_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")

        tr_noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")  # for training scheduler, fix as ddpm
        if self.args.eval_scheduler == "ddpm":
            noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
        elif self.args.eval_scheduler == "ddim":
            noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
        else:
            noise_scheduler = PNDMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
        feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_path, subfolder="feature_extractor")
        print(f"Loading pre-trained image_encoder from {args.pretrained_model_path}/image_encoder")
        clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_path, subfolder="image_encoder")
        print(f"Loading pre-trained vae from {args.pretrained_model_path}/vae")
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        print(f"Loading pre-trained unet from {self.args.pretrained_model_path}/unet")
        unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_path, subfolder="unet")

        # SOOHYUN EDIT START
        criterionGAN = GANLoss("lsgan")

        netD = PatchDiscriminator(
            input_nc=3,
            ndf=64,
            n_layers=3,
            norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False),
            no_antialias=False,
        )
        # EDIT END

        VGG = models.vgg19(pretrained=True).features
        VGG.to("cuda")
        for parameter in VGG.parameters():
            parameter.requires_grad_(False)

        if hasattr(noise_scheduler.config, "steps_offset") and noise_scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {noise_scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {noise_scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(self.noise_scheduler.config)
            new_config["steps_offset"] = 1
            noise_scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(noise_scheduler.config, "clip_sample") and noise_scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {noise_scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(noise_scheduler.config)
            new_config["clip_sample"] = False
            noise_scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(self.unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
            else:
                print("xformers is not available, therefore not enabled")

        # WT: initialize controlnet from the pretrained image variation SD model
        controlnet_pose = ControlNetModel.from_unet(unet=unet, args=self.args)

        if self.args.gradient_checkpointing:
            # controlnet_pose.enable_gradient_checkpointing()
            controlnet_pose.enable_gradient_checkpointing()
        # controlnet_unit = MultiControlNetModel_MultiHiddenStates([controlnet_pose, controlnet_pose])
        controlnet_unit = controlnet_pose

        self.tr_noise_scheduler = tr_noise_scheduler
        self.noise_scheduler = noise_scheduler
        self.vae = vae
        self.controlnet = controlnet_unit
        self.unet = unet

        self.criterionGAN = criterionGAN
        self.netD = netD

        self.VGG = VGG
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.clip_image_encoder = clip_image_encoder
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.metrics_accumulator = MetricsAccumulator()

        # self.ddpm_feature_extractor = FeatureExtractorDDPM(
        #     model = self.model,
        #     blocks = [5, 6, 7, 8, 12],
        #     input_activations = False,
        #     **unet.config
        # )

        self.drop_text_prob = args.drop_text
        self.device = torch.device("cpu")
        self.dtype = torch.float32

        self.scale_factor = self.args.scale_factor
        self.guidance_scale = self.args.guidance_scale
        self.controlnet_conditioning_scale = getattr(self.args, "controlnet_conditioning_scale", 1.0)
        self.controlnet_conditioning_scale_cond = getattr(self.args, "controlnet_conditioning_scale_cond", 1.0)
        self.controlnet_conditioning_scale_ref = getattr(self.args, "controlnet_conditioning_scale_ref", 1.0)

        if getattr(self.args, "combine_clip_local", None) and not getattr(
            self.args, "refer_clip_proj", None
        ):  # not use clip pretrained visual projection (but initialize from it)
            self.refer_clip_proj = torch.nn.Linear(
                clip_image_encoder.visual_projection.in_features,
                clip_image_encoder.visual_projection.out_features,
                bias=False,
            )
            self.refer_clip_proj.load_state_dict(clip_image_encoder.visual_projection.state_dict())
            self.refer_clip_proj.requires_grad_(True)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def init_ddpm(self):
        self.freeze_pretrained_part_in_ddpm()
        return

    def freeze_pretrained_part_in_ddpm(self):
        if self.args.freeze_unet:
            # b self.unet.eval()
            param_unfreeze_num = 0
            if self.args.unet_unfreeze_type == "crossattn-kv":
                for (
                    param_name,
                    param,
                ) in self.unet.named_parameters():  # only to set attn2 k, v to be requires_grad
                    if "transformer_blocks" not in param_name:
                        param.requires_grad_(False)
                    elif not ("attn2.to_k" in param_name or "attn2.to_v" in param_name):
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(True)
                        param_unfreeze_num += 1

            elif self.args.unet_unfreeze_type == "crossattn":
                for (
                    param_name,
                    param,
                ) in self.unet.named_parameters():  # only to set attn2 k, v to be requires_grad
                    if "transformer_blocks" not in param_name:
                        param.requires_grad_(False)
                    elif not "attn2" in param_name:
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(True)
                        param_unfreeze_num += 1

            elif self.args.unet_unfreeze_type == "transblocks":
                for param_name, param in self.unet.named_parameters():
                    if "transformer_blocks" not in param_name:
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(True)
                        param_unfreeze_num += 1

            elif self.args.unet_unfreeze_type == "all":
                for param_name, param in self.unet.named_parameters():
                    param.requires_grad_(True)
                    param_unfreeze_num += 1

            else:  # freeze all the unet
                print("Unmatch to any option, freeze all the unet")
                self.unet.eval()
                self.unet.requires_grad_(False)

            print(f"Mode [{self.args.unet_unfreeze_type}]: There are {param_unfreeze_num} modules in unet to be set as requires_grad=True.")

        self.vae.eval()
        self.clip_image_encoder.eval()
        self.vae.requires_grad_(False)
        self.clip_image_encoder.requires_grad_(False)
        if hasattr(self, "text_encoder"):
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)

    def to(self, *args, **kwargs):
        model_converted = super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.dtype = next(self.unet.parameters()).dtype
        self.clip_image_encoder.float()
        if hasattr(self, "text_encoder"):
            self.text_encoder.float()
        # self.refer_clip_proj.float()
        # self.vae.float()
        return model_converted

    def half(self, *args, **kwargs):
        super().half(*args, **kwargs)
        self.dtype = torch.float16
        self.clip_image_encoder.float()
        if hasattr(self, "text_encoder"):
            self.text_encoder.float()
        # self.refer_clip_proj.float()
        # self.vae.float()
        return

    def train(self, *args):
        super().train(*args)
        self.freeze_pretrained_part_in_ddpm()

    def image_encoder(self, image):
        b, c, h, w = image.size()
        latents = self.vae.encode(image).latent_dist.sample()
        latents = latents * self.scale_factor
        latents = latents.to(dtype=self.dtype)
        return latents

    def image_decoder(self, latents):
        latents = 1 / self.scale_factor * latents
        dec = self.vae.decode(latents).sample
        image = (dec / 2 + 0.5).clamp(0, 1)
        return image

    def forward(self, inputs):
        outputs = dict()

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.args.seed)
        inputs["generator"] = generator

        # Soohyun #
        ref_image = inputs["reference_img"]

        # #
        if self.training:
            assert self.args.stepwise_sample_depth <= 0
            # outputs = self.forward_train(inputs, outputs)
            outputs = self.forward_train_multicontrol(inputs, outputs)
        elif "enc_dec_only" in inputs and inputs["enc_dec_only"]:
            outputs = self.forward_enc_dec(inputs, outputs)
        else:
            assert self.args.stepwise_sample_depth <= 0
            # outputs = self.forward_sample(inputs, outputs)
            outputs = self.forward_sample_multicontrol(inputs, outputs)
        removed_key = inputs.pop("generator", None)
        return outputs

    def text_encode(
        self,
        prompt,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        negative_prompt=None,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(self.device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(self.device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !=" f" {type(prompt)}.")
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(self.device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        text_embeddings = text_embeddings.to(dtype=self.dtype)
        return text_embeddings

    def clip_encode_image_global(self, image, num_images_per_prompt=1, do_classifier_free_guidance=False):  # clip global feature
        dtype = next(self.clip_image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

        image = image.to(device=self.device, dtype=dtype)
        image_embeddings = self.clip_image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        return image_embeddings.to(dtype=self.dtype)

    def clip_encode_image_local(self, image, num_images_per_prompt=1, do_classifier_free_guidance=False):  # clip local feature
        dtype = next(self.clip_image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

        image = image.to(device=self.device, dtype=dtype)
        last_hidden_states = self.clip_image_encoder(image).last_hidden_state
        last_hidden_states_norm = self.clip_image_encoder.vision_model.post_layernorm(last_hidden_states)

        if self.args.refer_clip_proj:  # directly use clip pretrained projection layer
            image_embeddings = self.clip_image_encoder.visual_projection(last_hidden_states_norm)
        else:
            image_embeddings = self.refer_clip_proj(last_hidden_states_norm.to(dtype=self.dtype))
        # image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        return image_embeddings.to(dtype=self.dtype)

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().float().detach().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta=0.0):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, batch_size, num_channels_latents, height, width, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [torch.randn(shape, generator=generator[i], dtype=self.dtype) for i in range(batch_size)]
                latents = torch.cat(latents, dim=0).to(self.device)
            else:
                latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device=self.device, dtype=self.dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.noise_scheduler.init_noise_sigma
        return latents

    def get_features(self, image, model, layers=None):
        if layers is None:
            layers = {
                "0": "conv1_1",
                "5": "conv2_1",
                "10": "conv3_1",
                "19": "conv4_1",
                "21": "conv4_2",
                "28": "conv5_1",
                "31": "conv5_2",
            }
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features

    def feat_loss(self, x, x_t):
        self.VGG.cuda().to(self.device)
        x = x.half()
        x_t = x_t.half()
        x_features = self.get_features(x, self.VGG)
        x_t_features = self.get_features(x_t, self.VGG)

        loss = 0
        loss += torch.mean((x_features["conv4_2"] - x_t_features["conv4_2"]) ** 2)
        loss += torch.mean((x_features["conv5_2"] - x_t_features["conv5_2"]) ** 2)

        return loss

    def zecon_loss(self, x_in, y_in, t):
        loss = zecon_loss_direct(self.unet, x_in, y_in, torch.zeros_like(t, device=self.device))
        return loss.mean()

    def img_norm(self, image):
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)

        image = (image - mean) / std
        return image

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance,
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    def next_step(self, model_output: torch.FloatTensor, timestep: int, x: torch.FloatTensor, eta=0.0, verbose=False):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.ddim_inversion_scheduler.config.num_train_timesteps // self.ddim_inversion_scheduler.num_inference_steps, 999)
        alpha_prod_t = self.ddim_inversion_scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.ddim_inversion_scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.ddim_inversion_scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    @torch.no_grad()
    def invert(
        self, image: torch.Tensor, refer_latents, prompt="", num_inference_steps=50, guidance_scale=7.5, eta=0.0, return_intermediates=False, **kwds
    ):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size
        # text embeddings
        if guidance_scale > 1.0:
            text_input = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
            text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
            print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.image_encoder(image)
        start_latents = latents
        # print(latents)
        # exit()
        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.0:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # interative sampling
        self.ddim_inversion_scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.ddim_inversion_scheduler.timesteps))
        # print("attributes: ", self.ddim_inversion_scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]

        with torch.autocast("cuda"):
            for i, t in enumerate(tqdm(reversed(self.ddim_inversion_scheduler.timesteps), desc="DDIM Inversion")):
                if guidance_scale > 1.0:
                    model_inputs = torch.cat([latents] * 2)
                else:
                    model_inputs = latents

                noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample

                if guidance_scale > 1.0:
                    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

                # compute the previous noise sample x_t-1 -> x_t
                latents, pred_x0 = self.next_step(noise_pred, t, latents)
                latents_list.append(latents)
                pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents, start_latents

    def cond_fn_pix(self, gt_image, recon):  # x[0:1] = src, [1:2] = tgt
        with torch.enable_grad():
            gt_image = gt_image[:1].detach().requires_grad_(False)
            recon = recon[-1:].detach().requires_grad_(True)

            total_loss = 0

            if self.args.l_vgg != 0:
                loss_feature = self.feat_loss(
                    self.img_norm(gt_image.half().cuda().to(self.device)),
                    self.img_norm(recon.half().cuda().to(self.device)),
                )
                total_loss += loss_feature * self.args.l_vgg
            """ 
            zecon loss paper is based on pixel-level DDPM,
            so you need to think about how to apply it to latent-level DDPM.
            """
            if self.args.l_zecon != 0:
                x_in = recon
                y_t = self.noise_scheduler.q_sample(self.init_image, t)
                y_in = self.init_image * fac + y_t * (1 - fac)
                zecon_loss = self.zecon_loss(x_in, y_in, t) * self.args.l_zecon
                total_loss += zecon_loss * self.args.l_zecon

            return -(torch.autograd.grad(outputs=loss_feature, inputs=recon)[0])

    def cond_fn_latent(self, pred0_latents, start_latents):  # x[0:1] = src, [1:2] = tgt
        with torch.enable_grad():
            pred0_latents = pred0_latents.detach().requires_grad_(True)
            start_latents = start_latents.detach().requires_grad_(False)

            total_loss = 0

            if self.args.l_feat_mse != 0:
                feat_mse_loss = F.mse_loss(pred0_latents.float(), start_latents.float(), reduction="mean")
                total_loss += feat_mse_loss * self.args.l_feat_mse

            if self.args.l_vgg != 0:
                loss_feature = self.feat_loss(
                    self.img_norm(pred0_latents.half().cuda().to(self.device)),
                    self.img_norm(start_latents.half().cuda().to(self.device)),
                )
                total_loss += loss_feature * self.args.l_vgg
            if self.args.l_zecon != 0:
                """
                zecon loss paper is based on pixel-level DDPM,
                so you need to think about how to apply it to latent-level DDPM.
                """
                x_in = latents
                y_t = self.noise_scheduler.q_sample(self.init_image, t)
                y_in = self.init_image * fac + y_t * (1 - fac)
                zecon_loss = self.zecon_loss(x_in, y_in, t) * self.args.l_zecon
                total_loss += zecon_loss * self.args.l_zecon
            if self.args.l_feat_mse == 0 and self.args.l_vgg == 0 and self.args.l_zecon == 0:
                return 0
            return -(torch.autograd.grad(outputs=total_loss, inputs=pred0_latents)[0])

    @torch.no_grad()
    def forward_sample_multicontrol(self, inputs, outputs):
        gt_image = inputs["label_imgs"]
        ref_image = inputs["reference_img"]
        b, c, h, w = gt_image.size()

        do_classifier_free_guidance = self.guidance_scale > 1.0
        if self.args.combine_clip_local:
            refer_latents = self.clip_encode_image_local(ref_image, self.args.num_inf_images_per_prompt, do_classifier_free_guidance)
        else:
            refer_latents = self.clip_encode_image_global(ref_image, self.args.num_inf_images_per_prompt, do_classifier_free_guidance)

        """ ddim inversion via DisCo, for getting the x_t & x_0 """
        latents, start_latents = self.invert(gt_image, refer_latents)

        save_root = self.args.eval_save_filename
        save_root = os.path.join(save_root, "step-by-step")
        if not os.path.exists(save_root):
            os.makedirs(save_root, exist_ok=True)

        # str_images = self.image_decoder(start_latents)
        # save_image(str_images, "start_latents.png")
        # save_image(gt_image, "gt_image.png")
        # save_image(inputs["label_imgs"], "label_imgs.png")
        # save_image(inputs["cond_imgs"], "cond_imgs.png")
        # save_image(inputs["reference_img"], "reference_img.png")

        if self.args.ref_null_caption:  # test must use null caption
            text = inputs["input_text"]
            text = ["" for i in text]
            text_embeddings = self.text_encode(
                text,
                num_images_per_prompt=self.args.num_inf_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=None,
            )

        # Prepare conditioning image
        if self.controlnet_conditioning_scale:
            controlnet_conditioning_scale = self.controlnet_conditioning_scale
            controlnet_conditioning_scale_cond = self.controlnet_conditioning_scale_cond
            controlnet_conditioning_scale_ref = self.controlnet_conditioning_scale_ref
            controlnet_image = self.prepare_image(
                image=inputs["cond_imgs"].to(dtype=self.dtype),
                width=w,
                height=h,
                batch_size=b * self.args.num_inf_images_per_prompt,
                num_images_per_prompt=self.args.num_inf_images_per_prompt,
                device=self.device,
                dtype=self.controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )

        # Prepare timesteps
        self.noise_scheduler.set_timesteps(self.args.num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        gen_height = h
        gen_width = w
        generator = inputs["generator"]

        # latents = self.prepare_latents(
        #     b * self.args.num_inf_images_per_prompt,
        #     num_channels_latents,
        #     gen_height,
        #     gen_width,
        #     generator,
        #     latents=latents,  # seyeon
        # )

        if self.guidance_scale > 1.0:
            model_inputs = torch.cat([latents] * 2)
        else:
            model_inputs = latents

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator)

        # Denoising loop
        num_warmup_steps = len(timesteps) - self.args.num_inference_steps * self.noise_scheduler.order
        with self.progress_bar(total=self.args.num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                print("ttttttttttttttttttttttttttttttttttt\n")
                print(t)
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                if self.args.ref_null_caption:  # null caption input for ref controlnet path
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=controlnet_image,
                        conditioning_scale=controlnet_conditioning_scale_ref,
                        return_dict=False,
                    )
                else:
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=refer_latents,
                        controlnet_cond=controlnet_image,
                        conditioning_scale=controlnet_conditioning_scale_ref,
                        return_dict=False,
                    )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=refer_latents,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample.to(dtype=self.dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1, x_0
                latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                pred_0_latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).pred_original_sample

                str_images = self.image_decoder(pred_0_latents)
                save_image(str_images, os.path.join(save_root, f"{str(t.item())}pred_0_latents_bf_mse.png"))

                """ For computing pixel-level guidance"""
                # recon = self.decode_latents(pred_0_latents.to(self.device))
                # recon = torch.from_numpy(recon).to(self.device)
                # grad = self.cond_fn_pix(gt_image, recon)

                """ For computing latent-level guidance"""
                grad = self.cond_fn_latent(pred_0_latents, start_latents)  # (torch.Size([4, 4, 32, 32]), torch.Size([4, 4, 32, 32]))

                if grad is not 0:
                    alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
                    noise_pred = noise_pred - ((1 - alpha_prod_t) ** 0.5) * grad.detach()
                    latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    pred_0_latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).pred_original_sample

                    str_images = self.image_decoder(pred_0_latents)
                    save_image(str_images, os.path.join(save_root, f"{str(t.item())}pred_0_latents_af_mse.png"))

                # pdb.set_trace()

                ### seyeon added END ###

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.noise_scheduler.order == 0):
                    progress_bar.update()
        # Post-processing
        gen_img = self.image_decoder(latents)

        outputs["logits_imgs"] = gen_img
        return outputs

    @torch.no_grad()
    def forward_enc_dec(self, inputs, outputs):
        image = inputs["label_imgs"]
        latent = self.image_encoder(image)
        gen_img = self.image_decoder(latent)
        outputs["logits_imgs"] = gen_img
        return outputs

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}.")

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs


def inner_collect_fn(args, inputs, outputs, log_dir, global_step, eval_save_filename="eval_visu"):
    rank = get_rank()
    if rank == -1:
        splice = ""
    else:
        splice = "_" + str(rank)
    if global_step <= 0:
        eval_log_dir = os.path.join(log_dir, eval_save_filename)
    else:
        eval_log_dir = os.path.join(log_dir, "eval_step_%d" % (global_step))
    ensure_directory(eval_log_dir)

    gt_save_path = os.path.join(eval_log_dir, "gt")
    ensure_directory(gt_save_path)
    pred_save_path = os.path.join(
        eval_log_dir,
        f"pred_gs{args.guidance_scale}_scale-cond{args.controlnet_conditioning_scale_cond}-ref{args.controlnet_conditioning_scale_ref}",
    )
    ensure_directory(pred_save_path)
    cond_save_path = os.path.join(eval_log_dir, "cond")
    ensure_directory(cond_save_path)
    ref_save_path = os.path.join(eval_log_dir, "ref")
    ensure_directory(ref_save_path)
    gt_masked_save_path = os.path.join(eval_log_dir, "gt_masked")
    ensure_directory(gt_masked_save_path)

    synchronize()
    if rank in [-1, 0]:
        logger.warning(eval_log_dir)

        # Save Model Setting
        type_output = [
            int,
            float,
            str,
            bool,
            tuple,
            dict,
            type(None),
        ]
        setting_output = {item: getattr(args, item) for item in dir(args) if type(getattr(args, item)) in type_output and not item.startswith("__")}
        data2file(setting_output, os.path.join(eval_log_dir, "Model_Setting.json"))

    dl = {**inputs, **{k: v for k, v in outputs.items() if k.split("_")[0] == "logits"}}
    ## WT DEBUG
    # print('just for debug')
    # if 'cond_img_pose' in dl:
    #     del dl['cond_img_pose']
    #     del dl['cond_img_attr']
    ld = dl2ld(dl)

    l = ld[0]["logits_imgs"].shape[0]

    for _, sample in enumerate(ld):
        _name = "nuwa"
        if "input_text" in sample:
            _name = sample["input_text"][:200]
        if "img_key" in sample:
            _name = sample["img_key"]
        char_remov = [os.sep, ".", "/"]
        for c in char_remov:
            _name = _name.replace(c, "")
        prefix = "_".join(_name.split(" "))
        # prefix += splice
        # postfix = '_' + str(round(time.time() * 10))
        postfix = ""

        if "label_imgs" in sample:
            image = sample["label_imgs"]
            image = (image / 2 + 0.5).clamp(0, 1)
            image = tensor2pil(image)[0]
            try:
                image.save(os.path.join(gt_save_path, prefix + postfix + ".png"))
            except Exception as e:
                print(f"some errors happened in saving label_imgs: {e}")
        if "logits_imgs" in sample:
            try:
                image = tensor2pil(sample["logits_imgs"])[0]
                image.save(os.path.join(pred_save_path, prefix + postfix + ".png"))
            except Exception as e:
                print(f"some errors happened in saving logits_imgs: {e}")
        if "cond_imgs" in sample and sample["cond_imgs"] is not None:  # pose
            image = tensor2pil(sample["cond_imgs"])[0]
            try:
                image.save(os.path.join(cond_save_path, prefix + postfix + ".png"))
            except Exception as e:
                print(f"some errors happened in saving logits_imgs: {e}")
        if "reference_img" in sample:
            image = sample["reference_img"]
            image = (image / 2 + 0.5).clamp(0, 1)
            image = tensor2pil(image)[0]
            try:
                image.save(os.path.join(ref_save_path, prefix + postfix + ".png"))
            except Exception as e:
                print(f"some errors happened in saving label_imgs: {e}")
        if "foreground_label_img" in sample:
            image = sample["foreground_label_img"]
            image = (image / 2 + 0.5).clamp(0, 1)
            image = tensor2pil(image)[0]
            try:
                image.save(os.path.join(gt_masked_save_path, prefix + postfix + ".png"))
            except Exception as e:
                print(f"some errors happened in saving label_imgs: {e}")
    return gt_save_path, pred_save_path


def tensor2pil(images):
    # c, h, w
    images = images.cpu().permute(1, 2, 0).float().numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


# EDIT START
def compute_D_loss(fake_B, real_B, netD, criterionGAN):
    """Calculate GAN loss for the discriminator"""
    fake = fake_B.detach()
    # fake = self.fake_pin.detach()
    # Fake; stop backprop to the generator by detaching fake_B

    pred_fake = netD(fake)
    loss_D_fake = criterionGAN(pred_fake, False).mean()

    # Real
    real_B.requires_grad_()
    pred_real = netD(real_B)  # , self.netG.ContentEncoder.reflectionpad_1, self.netG.ContentEncoder.deform_conv_1, pano=False, pin=True)

    # self.loss_reg = self.r1_reg(self.pred_real, self.real_B)

    loss_D_real = criterionGAN(pred_real, True)
    loss_D_real = loss_D_real.mean()

    # combine loss and calculate gradients
    # self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
    loss_D = (loss_D_fake + loss_D_real) * 0.5
    return loss_D


def compute_G_loss(fake_B, netD, criterionGAN):
    """Calculate GAN loss for the discriminator"""
    fake = fake_B.detach()
    # fake = self.fake_pin.detach()
    # Fake; stop backprop to the generator by detaching fake_B

    pred_fake = netD(fake)
    loss_D_fake = criterionGAN(pred_fake, True).mean()

    # combine loss and calculate gradients
    # self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
    loss_G = loss_D_fake
    return loss_G


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp", "nonsaturating"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        bs = prediction.size(0)
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == "nonsaturating":
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        return loss


# EDIT END
