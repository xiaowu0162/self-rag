import os
import sys
import json
from tqdm import tqdm
from datetime import datetime
import subprocess
import numpy as np


in_file = sys.argv[1]
data = [json.loads(line) for line in open(in_file).readlines()]

working_dir = f'temp_dir_factscore_{datetime.now().strftime("%Y%m%d-%H%M")}/'
os.makedirs(working_dir)

command = 'python -m factscore.factscorer --input_path {} --model_name retrieval+ChatGPT --data_dir /local2/diwu/selfrag_model_cache/factscore/ --cache_dir /local2/diwu/selfrag_model_cache/factscore/ --openai_key openai.key --verbose'

all_scores = []
for i, entry in tqdm(enumerate(data)):
    tempfile = working_dir + f'/{i}.json'
    with open(tempfile, 'w') as f:
         print(json.dumps(entry), file=f)

    cur_command = command.format(tempfile).split()
    # result = subprocess.run(cur_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = subprocess.run(cur_command, capture_output=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    result_lines = result.stderr.decode('utf-8').split('\n')
    result_lines = [line for line in result_lines if 'FActScore = ' in line]
    try:
        result_line = result_lines[0]
        cur_result = float(result_line.split('FActScore = ')[-1].strip('%'))
    except:
        cur_result = 'n/a'
    all_scores.append(cur_result)
    print('FactScore =', cur_result, 'Moving avg =',
          np.mean([x for x in all_scores if x != 'n/a']).item())
        
        
out_file = in_file + '.eval.log'
with open(out_file, 'w') as f:
    for score in all_scores:
        print(score, file=f)
