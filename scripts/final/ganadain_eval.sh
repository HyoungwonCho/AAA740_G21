for WEATHER in sunny2night night2sunny cloudy2sunny
do
    CUDA_VISIBLE_DEVICES=0 python make_images_CityScape.py \
    --cf config/ref_attn_clip_combine_controlnet_attr_pretraining/Init_config_eval.py \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --root_dir visu/ --log_dir sampling/ \
    --max_eval_samples 1000 \
    --pretrained_model /media/dataset/14499.pth \
    --eval_visu \
    --eval_save_filename /media/dataset/eval_visualization/INIT/${WEATHER} \
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
    --combine_clip_local --combine_use_mask 
done
