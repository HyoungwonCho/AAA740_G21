CUDA_VISIBLE_DEVICES=1 python finetune_sdm_INIT.py \
--cf /home/cvlab10/project/disco-i2i_GAN/config/ref_attn_clip_combine_controlnet/config_BDD.py \
--do_train --root_dir /media/data/run_test/ \
--pretrained_model_dreambooth /media/data/run_test/bdd/dreambooth2/999.pth/mp_rank_00_model_states.pt \
--log_dir bdd/seg_edge_transblock_dreambooth_controlnet \
--local_train_batch_size 32 \
--local_eval_batch_size 32 \
--epochs 400 --deepspeed \
--eval_step 500 --save_step 500 \
--gradient_accumulate_steps 1 \
--learning_rate 2e-4 --fix_dist_seed --loss_target "noise" \
--refer_sdvae \
--ref_null_caption False \
--combine_clip_local --combine_use_mask \
--conds "masks" \
--max_eval_samples 100 --node_split_sampler 0

# --unet_unfreeze_type "all" \