import os
import sys
import json
from tqdm import tqdm
from datetime import datetime
import subprocess
import numpy as np


def average_fact_score(entries):
    n_facts = sum([x['n_facts'] for x in entries])
    scoresum = sum([x['factscore'] * x['n_facts'] for x in entries if x['factscore'] != 'n/a'])
    return scoresum / n_facts


in_file = sys.argv[1]
out_file = in_file + '.eval.log'
data = [json.loads(line) for line in open(in_file).readlines()]

working_dir = f'temp_dir_factscore_{datetime.now().strftime("%Y%m%d-%H%M")}/'
os.makedirs(working_dir)

command = 'python -m factscore.factscorer --input_path {} --model_name retrieval+ChatGPT --data_dir /local2/diwu/selfrag_model_cache/factscore/ --cache_dir /local2/diwu/selfrag_model_cache/factscore/ --openai_key openai.key --verbose'

all_scores = []
with open(out_file, 'w') as out_f:
    for i, entry in tqdm(enumerate(data)):
        tempfile = working_dir + f'/{i}.json'
        with open(tempfile, 'w') as f:
            print(json.dumps(entry), file=f)

        cur_command = command.format(tempfile).split()
        result = subprocess.run(cur_command, capture_output=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        result_lines = result.stderr.decode('utf-8').split('\n')
        try:
            result_line = result_lines[0]
            cur_score = float([line for line in result_lines if 'FActScore = ' in line][0].split('FActScore = ')[-1].strip('%'))
            cur_nfacts = float([line for line in result_lines if '# Atomic facts' in line][0].split('# Atomic facts per valid response = ')[-1].strip())
        except:
            cur_nfacts = 0
            cur_score = 'n/a'
        cur_scoring_entry = {'n_facts': cur_nfacts, 'factscore': cur_score}
        all_scores.append(cur_scoring_entry)
        print('FactScore =', cur_score, 'n_facts =', cur_nfacts,
            'Moving avg (doc) = {:.02f}'.format(np.mean([x['factscore'] for x in all_scores if x != 'n/a']).item()), 
            'Moving avg (fact) = {:.02f}'.format(average_fact_score(all_scores)))
        print(json.dumps(cur_scoring_entry), file=out_f)
