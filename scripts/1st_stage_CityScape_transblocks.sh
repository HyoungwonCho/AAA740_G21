CUDA_VISIBLE_DEVICES=1,2,3 mpirun -np 3 python finetune_sdm_CityScape.py \
--cf config/ref_attn_clip_combine_controlnet_attr_pretraining/CityScape_config.py \
--do_train --root_dir /media/data/run_test/ \
--local_train_batch_size 12 \
--local_eval_batch_size 12 \
--log_dir /media/data/run_test/CityScape_cont/cont_0816 \
--epochs 40 --deepspeed \
--eval_step 200 --save_step 200 \
--gradient_accumulate_steps 1 \
--learning_rate 2e-4 --fix_dist_seed --loss_target "noise" \
--unet_unfreeze_type "transblocks" \
--refer_sdvae \
--ref_null_caption False \
--combine_clip_local --combine_use_mask \
--conds "masks" \
--max_eval_samples 2000 --node_split_sampler 0 \
--resume 
