CUDA_VISIBLE_DEVICES=1,2,3 mpirun -np 3 python finetune_sdm_BDD.py \
--pretrained_model /media/data/run_test/bdd/seg_edge2/5499.pth \
--cf config/ref_attn_clip_combine_controlnet/config_BDD.py \
--do_train --root_dir /media/data/run_test/ \
--log_dir bdd/seg_edge3 \
--eval_before_train True \
--img_size 256 \
--local_train_batch_size 32 \
--local_eval_batch_size 32 \
--epochs 10000 --deepspeed \
--eval_step 200 --save_step 500 \
--gradient_accumulate_steps 1 \
--learning_rate 2e-5 --fix_dist_seed --loss_target "noise" \
--unet_unfreeze_type "all" \
--ref_null_caption False \
--combine_use_mask \
--conds "masks" \
--max_eval_samples 1000 --node_split_sampler 0 \
--combine_clip_local 

#--pretrained_model /media/dataset/run_test/Init/edgedepth_condition3/3499.pth \
