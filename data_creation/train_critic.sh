#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONIOENCODING=utf-8;
HOME_DIR=`realpath ..`
export PYTHONPATH=${PYTHONPATH}:${HOME_DIR}
# export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256'


add_reflection_trigger=true


torchrun --nproc_per_node=4 --master_port=2570 \
	 train_special_tokens.py \
	 --use_special_token \
	 --add_reflection_trigger ${add_reflection_trigger} \
	 --model_name_or_path NousResearch/Llama-2-7b-hf \
	 --data_path critic_training_data_gpt4_reward_all_0813_train_4k_retrieval.json \
	 --bf16 True \
	 --output_dir /local2/diwu/selfrag_model_cache/20240111_selfrag_critic_4kretrievaldata_reflectiontrigger${add_reflection_trigger}/ \
	 --num_train_epochs 3  \
	 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
	 --gradient_accumulation_steps 32 \
	 --evaluation_strategy "no" \
	 --save_strategy "steps" \
	 --save_steps 100 \
	 --save_total_limit 1 \
	 --learning_rate 2e-5 \
	 --weight_decay 0. \
	 --warmup_ratio 0.01 \
	 --lr_scheduler_type "cosine" \
	 --logging_steps 10 \
	 --fsdp "full_shard auto_wrap"
