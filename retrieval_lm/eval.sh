#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

export task=$1
export model=$2  
export exp=$3  # no-rag, always-rag, adaptive-rag, adaptive-rag-greedy
export threshold=${4:-"0.2"}

data_root=`realpath ./eval_data/`
output_root=`realpath ../results/`

declare -A MODEL_ZOO
MODEL_ZOO["selfrag-7b"]="selfrag/selfrag_llama2_7b"
MODEL_ZOO["selfrag-13b"]="selfrag/selfrag_llama2_13b"

model_name=${MODEL_ZOO["$model"]}


# task-specific params
gen_length=100
max_depth=2
beam_width=2
metric=match
if [[ $task == "popqa" ]]; then
    script="run_short_form.py"
    use_task='qa'
    ndocs=10
    prompt_file="${data_root}/popqa_longtail_w_gs.jsonl"
elif [[ $task == "triviaqa" ]]; then
    script="run_short_form.py"
    use_task='qa'
    ndocs=10
    prompt_file="${data_root}/triviaqa_test.jsonl"
elif [[ $task == "fever" ]]; then
    script="run_short_form.py"
    use_task='fever'
    ndocs=5
    gen_length=50
    prompt_file="${data_root}/health_claims_processed.jsonl"
elif [[ $task == "arc_c" ]]; then
    script="run_short_form.py"
    use_task='arc_c'
    ndocs=5
    gen_length=50
    prompt_file="${data_root}/arc_challenge_processed.jsonl"
elif [[ $task == "bio" ]]; then
    script="run_long_form_static.py"
    use_task='factscore'
    ndocs=5
    gen_length=300
    max_depth=7
    beam_width=2
    metric="n/a"
    prompt_file="${data_root}/factscore_unlabeled_alpaca_13b_retrieval.jsonl"
elif [[ $task == "asqa" ]]; then
    script="run_long_form_static.py"
    use_task='asqa'
    ndocs=5
    gen_length=300
    max_depth=7
    beam_width=2
    metric="n/a"
    prompt_file="${data_root}/asqa_eval_gtr_top100.json"
else
    echo "Unsupported task: ${task}"
fi

# mode
if [[ $exp == "no-rag" ]]; then
    mode='no_retrieval'
    threshold_flag=""
    outfile="${model}.json"
elif [[ $exp == "always-rag" ]]; then
    mode='always_retrieve'
    threshold_flag="--threshold 0"
    outfile="${model}.json"
elif [[ $exp == "adaptive-rag" ]]; then
    mode='adaptive_retrieval'
    threshold_flag="--threshold ${threshold}"
    outfile="${model}-t${threshold}.json"
elif [[ $exp == "adaptive-rag-greedy" ]]; then
    mode='adaptive_retrieval'
    threshold_flag=""
    outfile="${model}.json"
else
    echo "Unsupported exp: ${exp}"
fi

output_dir=$output_root/$task/$exp/
mkdir -p $output_dir


python ${script}   \
       --model_name ${model_name} \
       --input_file ${prompt_file} \
       --max_new_tokens ${gen_length} \
       --mode ${mode} ${threshold_flag} \
       --output_file ${output_dir}/${outfile} \
       --metric ${metric} \
       --ndocs ${ndocs} \
       --max_depth ${max_depth} \
       --beam_width ${beam_width} \
       --use_groundness \
       --use_utility \
       --use_seqscore \
       --store_retrieval_prob \
       --task ${use_task}
