#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

export task=$1
export model=$2  
export exp=$3  # no-rag, always-rag

data_root=`realpath ./eval_data/`
output_root=`realpath ../results/`

declare -A MODEL_ZOO
MODEL_ZOO["llama-2-7b"]="NousResearch/Llama-2-7b-hf"
MODEL_ZOO["llama-2-13b"]="NousResearch/Llama-2-13b-hf"
MODEL_ZOO["alpaca-7b"]="/local2/diwu/selfrag_model_cache/Llama-2-7b-alpaca-cleaned"   # NEU-HAI/Llama-2-7b-alpaca-cleaned

model_name=${MODEL_ZOO["$model"]}


# task 
if [[ $task == "popqa" ]]; then
    use_task='qa'
    prompt_file="${data_root}/popqa_longtail_w_gs.jsonl"
else
    echo "Unsupported task: ${task}"
fi


# mode
if [[ $exp == "no-rag" ]]; then
    mode='vanilla'
    prompt_name='prompt_no_input'
elif [[ $exp == "always-rag" ]]; then
    mode='retrieval'
    prompt_name='prompt_no_input_retrieval'
else
    echo "Unsupported exp: ${exp}"
fi

output_dir=$output_root/$task/$exp/
mkdir -p $output_dir

gen_length=100

python run_baseline_lm.py \
       --mode ${mode} \
       --model_name ${model_name} \
       --input_file ${prompt_file} \
       --max_new_tokens ${gen_length} \
       --metric match \
       --result_fp ${output_dir}/${model}.json \
       --task ${use_task} \
       --prompt_name "prompt_no_input"

