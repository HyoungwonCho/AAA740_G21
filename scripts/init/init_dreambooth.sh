CUDA_VISIBLE_DEVICES=1,2,3 mpirun -np 3 python finetune_sdm_BDD.py \
--cf config/ref_attn_clip/config_INIT.py \
--eval_before_train False \
--do_train --root_dir /media/data/run_test/ \
--log_dir init/dreambooth2 \
--local_train_batch_size 32 \
--local_eval_batch_size 32 \
--epochs 40 --deepspeed \
--eval_step 500 --save_step 500 \
--gradient_accumulate_steps 1 \
--learning_rate 2e-4 --fix_dist_seed --loss_target "noise" \
--unet_unfreeze_type "all" \
--refer_sdvae \
--ref_null_caption False \
--combine_use_mask \
--conds "masks" \
--max_eval_samples 100 --node_split_sampler 0


# --combine_clip_local