#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

export task=$1
export model=$2  
export exp=$3  # no-rag, always-rag, adaptive-rag, adaptive-rag-greedy

data_root=`realpath ./eval_data/`
output_root=`realpath ../results/`

declare -A MODEL_ZOO
MODEL_ZOO["selfrag-7b"]="selfrag/selfrag_llama2_7b"
MODEL_ZOO["selfrag-13b"]="selfrag/selfrag_llama2_13b"

model_name=${MODEL_ZOO["$model"]}


# task 
if [[ $task == "popqa" ]]; then
    script="run_short_form.py"
    use_task='qa'
    prompt_file="${data_root}/popqa_longtail_w_gs.jsonl"
else
    echo "Unsupported task: ${task}"
fi


# mode
if [[ $exp == "no-rag" ]]; then
    mode='no_retrieval'
    threshold_flag="--threshold 0"
elif [[ $exp == "always-rag" ]]; then
    mode='always_retrieve'
    threshold_flag="--threshold 0"
elif [[ $exp == "adaptive-rag" ]]; then
    mode='adaptive_retrieval'
    threshold_flag="--threshold 0.2"
elif [[ $exp == "adaptive-rag-greedy" ]]; then
    mode='adaptive_retrieval'
    threshold_flag=""
else
    echo "Unsupported exp: ${exp}"
fi

output_dir=$output_root/$task/$exp/
mkdir -p $output_dir

gen_length=100

python ${script}   \
       --model_name ${model_name} \
       --input_file ${prompt_file} \
       --max_new_tokens ${gen_length} \
       --mode ${mode} ${threshold_flag} \
       --output_file ${output_dir}/${model}.json \
       --metric match \
       --ndocs 5 \
       --use_groundness \
       --use_utility \
       --use_seqscore \
       --task ${use_task}
