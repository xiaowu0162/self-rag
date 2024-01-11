import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams


PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    # "prompt_no_input": (
    #     "### Instruction:\n{instruction}\n\n### Response:"
    # ),
    "prompt_input_reflection_trigger": (
        "{input}{reflection_trigger}"
    ),
}


data_dir = '/home/diwu/ralm/self-rag/data_creation/'
critic_data = json.load(open(data_dir + '/critic_training_data_gpt4_reward_all_0813_train.json'))
critic_tasks = set([x['task'] for x in critic_data])
critic_task2data = {t: [x for x in critic_data if x['task'] == t] for t in critic_tasks} 

model = '/local2/diwu/selfrag_model_cache/20240105_selfrag_critic_alldata/checkpoint-1000/'   # selfrag/selfrag_llama2_7b selfrag/self_rag_critic
model_short = 'selfrag-critic-0105ckpt-alldata'   # selfrag-7b selfrag-critic
out_file = f'critic_data_selective_probs_{model_short}.txt'
out_file_greedy = f'critic_data_greedy_{model_short}.txt'

use_critic_instructions = True
if model == 'selfrag/selfrag_llama2_7b':
    assert not use_critic_instructions
elif model == 'selfrag/self_rag_critic':
    assert use_critic_instructions

tokenizer = AutoTokenizer.from_pretrained(model)
model = LLM(model=model, download_dir='/home/diwu/ralm/self-rag/retrieval_lm/.cache', dtype='half', tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2, logprobs=len(tokenizer))

retrieval_tokens_names = ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"]
ret_tokens = {token: tokenizer.convert_tokens_to_ids(token) for token in retrieval_tokens_names}

with open(out_file, 'w') as out_f, open(out_file_greedy, 'w') as out_f_greedy:
    for entry in tqdm(critic_task2data['retrieval']):
        if use_critic_instructions:
            preds = model.generate([PROMPT_DICT['prompt_input'].format_map(entry)], sampling_params, use_tqdm=False)
        else:
            preds = model.generate([entry['input']], sampling_params, use_tqdm=False)
        # print(entry)
        pred_token_ids = preds[0].outputs[0].token_ids
        pred_log_probs = preds[0].outputs[0].logprobs
        score_dict = {}
        for tok, id in ret_tokens.items():
            if id not in pred_log_probs[0]:
                score_dict[tok] = -100
            prob = pred_log_probs[0][id]
            score_dict[tok] = float(prob)
        retrieval_prob = score_dict["[Retrieval]"] / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"])
        
        if score_dict["[Retrieval]"] > score_dict["[No Retrieval]"]:
            greedy_label = '[Retrieval]'
        else:
            greedy_label = '[No Retrieval]'

        print(retrieval_prob, file=out_f, flush=True)
        print(greedy_label, file=out_f_greedy, flush=True)
        print(score_dict["[Retrieval]"], score_dict["[No Retrieval]"], retrieval_prob, greedy_label)
    