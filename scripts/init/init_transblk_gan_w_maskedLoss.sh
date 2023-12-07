CUDA_VISIBLE_DEVICES=4,5,6,7 mpirun -np 4  python finetune_sdm_INIT_gan.py \
--cf config/ref_attn_clip_combine_controlnet/config_INIT_gan_w_mask.py \
--do_train --root_dir /media/data1/run_test/ \
--log_dir Init/edge_gan_maskedLoss \
--eval_before_train False \
--img_size 256 \
--local_train_batch_size 14 \
--local_eval_batch_size 14 \
--epochs 50 --deepspeed \
--eval_step 200 --save_step 500 \
--gradient_accumulate_steps 1 \
--learning_rate 2e-5 --fix_dist_seed --loss_target "noise" \
--unet_unfreeze_type "transblocks" \
--ref_null_caption False \
--combine_use_mask \
--conds "masks" \
--max_eval_samples 1000 --node_split_sampler 0 \
--combine_clip_local 

#--pretrained_model /media/dataset/run_test/Init/edgedepth_condition3/3499.pth \
#,5,6,7 mpirun -np 4 