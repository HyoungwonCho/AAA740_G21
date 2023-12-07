WEATHER=rainy

CUDA_VISIBLE_DEVICES=1 python make_images_CityScape.py \
--cf config/ref_attn_clip_combine_controlnet_attr_pretraining/Init_config_eval.py \
--num_inference_steps 50 \
--guidance_scale 7.5 \
--root_dir visu/ --log_dir sampling/ \
--max_eval_samples 100000 \
--pretrained_model /media/data/run_test/Init/segedge/8999.pth \
--eval_visu \
--eval_save_filename ./eval_visualization_hw/INIT/${WEATHER} \
--eval_before_train False \
--local_train_batch_size 32 \
--local_eval_batch_size 32 \
--epochs 20 --deepspeed \
--eval_step 500 --save_step 500 \
--gradient_accumulate_steps 1 \
--refer_sdvae \
--ref_null_caption False \
--combine_use_mask 