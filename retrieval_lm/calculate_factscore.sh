OUTPUT_FILE=$1

python -m factscore.factscorer\
       --input_path ${OUTPUT_FILE} \
       --model_name retrieval+ChatGPT \
       --data_dir /local2/diwu/selfrag_model_cache/factscore/ \
       --cache_dir /local2/diwu/selfrag_model_cache/factscore/ \
       --openai_key openai.key \
       --verbose
