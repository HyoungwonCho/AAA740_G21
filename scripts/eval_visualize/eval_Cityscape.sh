EXP_NAME=GanLoss_and_ContentLoss_adain
COND_IMG=source2.png

CUDA_VISIBLE_DEVICES=5 python make_images_CityScape.py \
--cf config/ref_attn_clip_combine_controlnet_attr_pretraining/CityScape_eval_config.py \
--num_inference_steps 50 \
--guidance_scale 7.5 \
--root_dir visu/ --log_dir sampling/ \
--max_eval_samples 1000 \
--pretrained_model /media/data/run_test/CityScape_cont/0910_adain_resum/22199.pth \
--eval_visu \
--eval_save_filename ./eval_visualization/CityScape/${EXP_NAME} \
--eval_before_train False \
--local_train_batch_size 32 \
--local_eval_batch_size 32 \
--epochs 20 --deepspeed \
--eval_step 500 --save_step 500 \
--gradient_accumulate_steps 1 \
--learning_rate 2e-4 --fix_dist_seed --loss_target "noise" \
--unet_unfreeze_type "all" \
--refer_sdvae \
--ref_null_caption False \
--combine_clip_local --combine_use_mask \
--use_adain True \
--only_self_attention False \



#--fixed_condition_root /media/data/i2i_dataset/Syntheia/grsam_eval_masked/${COND_IMG} \
