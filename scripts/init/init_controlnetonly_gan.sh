CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -np 8 python finetune_sdm_INIT_gan.py \
--cf config/ref_attn_clip_combine_controlnet/config_INIT_gan.py \
--do_train --root_dir /media/data/run_test/ \
--log_dir Init/edgedepth_gan_controlnetONLY \
--eval_before_train False \
--img_size 256 \
--local_train_batch_size 14 \
--local_eval_batch_size 40 \
--epochs 50 --deepspeed \
--eval_step 250 --save_step 500 \
--gradient_accumulate_steps 1 \
--learning_rate 2e-5 --fix_dist_seed --loss_target "noise" \
--ref_null_caption False \
--combine_use_mask \
--conds "masks" \
--max_eval_samples 1000 --node_split_sampler 0 \
--pretrained_model_dreambooth /media/data/run_test/Init/dreambooth2/17499.pth/mp_rank_00_model_states.pt 
#--pretrained_model /media/dataset/run_test/Init/edgedepth_condition3/3499.pth \