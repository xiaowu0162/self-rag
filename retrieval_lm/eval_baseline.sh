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


# task-specific params
gen_length=100
if [[ $task == "popqa" ]]; then
    use_task='qa'
    ndocs=10
    prompt_file="${data_root}/popqa_longtail_w_gs.jsonl"
    instruction=""
elif [[ $task == "triviaqa" ]]; then
    use_task='qa'
    ndocs=10
    prompt_file="${data_root}/triviaqa_test.jsonl"
    instruction=""
elif [[ $task == "fever" ]]; then
    use_task='fever'
    ndocs=5
    gen_length=50
    prompt_file="${data_root}/health_claims_processed.jsonl"
    instruction="Is the following statement correct or not? Say true if it is correct; otherwise say false."
elif [[ $task == "arc_c" ]]; then
    use_task='arc_c'
    ndocs=5
    gen_length=50
    prompt_file="${data_root}/arc_challenge_processed.jsonl"
    instruction="Given four answer candidates, A, B, C and D, choose the best answer choice."
elif [[ $task == "asqa" ]]; then
    use_task='asqa'
    ndocs=5
    gen_length=300
    prompt_file="${data_root}/asqa_eval_gtr_top100.json"
    instruction="Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."
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

python run_baseline_lm.py \
       --mode ${mode} \
       --top_n ${ndocs} \
       --model_name ${model_name} \
       --input_file ${prompt_file} \
       --instruction "${instruction}" \
       --max_new_tokens ${gen_length} \
       --metric match \
       --result_fp ${output_dir}/${model}.json \
       --task ${use_task} \
       --prompt_name ${prompt_name}

