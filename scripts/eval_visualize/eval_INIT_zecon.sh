WEATHER=sunny

CUDA_VISIBLE_DEVICES=5 python make_images_CityScape.py \
--cf config/ref_attn_clip_combine_controlnet_attr_pretraining/Init_config_eval_sampling_zecon.py \
--l_zecon 0 \
--l_vgg 0 \
--l_feat_mse 0 \
--num_inference_steps 50 \
--guidance_scale 7.5 \
--root_dir visu/ --log_dir sampling/ \
--max_eval_samples 1000 \
--pretrained_model /media/data/run_test/Init/1st_trial/11999.pth \
--eval_visu \
--eval_save_filename ./eval_visualization/INIT_zecon/2_${WEATHER} \
--eval_before_train False \
--local_train_batch_size 4 \
--local_eval_batch_size 4 \
--epochs 20 --deepspeed \
--eval_step 500 --save_step 500 \
--gradient_accumulate_steps 1 \
--learning_rate 2e-4 --fix_dist_seed --loss_target "noise" \
--unet_unfreeze_type "all" \
--refer_sdvae \
--ref_null_caption False \
--combine_clip_local --combine_use_mask \
