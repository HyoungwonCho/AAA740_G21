for WEATHER in sunny2rainy
do
    CUDA_VISIBLE_DEVICES=3 python make_images_CityScape.py \
    --cf config/ref_attn_clip_combine_controlnet/config_INIT_multi_metric.py \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --root_dir visu/ --log_dir sampling/ \
    --max_eval_samples 10000000 \
    --pretrained_model /media/dataset/run_test_hw/Init/seg_edge_depth_resume/4999.pth \
    --eval_visu \
    --eval_class ${WEATHER} \
    --eval_save_filename /media/dataset/eval_visualization/INIT/baseline/${WEATHER} \
    --eval_before_train False \
    --local_train_batch_size 32 \
    --local_eval_batch_size 32 \
    --epochs 20 --deepspeed \
    --eval_step 500 --save_step 500 \
    --gradient_accumulate_steps 1 \
    --learning_rate 2e-4 --fix_dist_seed --loss_target "noise" \
    --refer_sdvae \
    --ref_null_caption False \
    --combine_clip_local --combine_use_mask 
done
