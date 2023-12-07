CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 python finetune_sdm_INIT.py \
--cf config/ref_attn_clip/config_INIT.py \
--eval_before_train True \
--do_train --root_dir /media/data/run_test/ \
--log_dir Init/dreambooth_refNull2 \
--pretrained_model_dreambooth /media/data/run_test/Init/dreambooth_refNull/4999.pth/mp_rank_00_model_states.pt \
--local_train_batch_size 40 \
--local_eval_batch_size 40 \
--epochs 40 --deepspeed \
--drop_ref 0.3 \
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