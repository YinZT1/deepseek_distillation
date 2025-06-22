import torch
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from lsrl import patch_qwen2_for_multi_gpus

model_dir = '/home/dell/yzt/qwen2.5-14b-instruct'
data_dir = '/home/dell/yzt/train.json'
output_dir = './res.json'
print('tokenizer loading ... ')
tokenizer = AutoTokenizer.from_pretrained(model_dir)
device_count = torch.cuda.device_count()
device = list(range(device_count))
print('model loading ...')
model = AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype = torch.bfloat16)
patch_qwen2_for_multi_gpus(model,device)
model.eval()
print('read data ...')
problems = []
with open(data_dir,'r') as f:
    i = 0
    for line in f:
        if i > 10 :
            break
        data = json.loads(line.strip())
        problems.append(data['problem'])
        i += 1

print('inference started')
with torch.no_grad(),open(output_dir,'w',encoding = 'utf-8') as f:
    for problem in tqdm(problems,desc = "producing answer ..."):
        prompt_txt = f"<s>User:{problem}\nAssistant:"
        inputs = tokenizer(prompt_txt,return_tensors = 'pt')
        inputs = inputs.to(f"cuda:{device[0]}")
        outputs = model.generate(
            **inputs,
            max_new_tokens = 1024,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id,
            do_sample = False,
        )
        input_len = inputs.input_ids.shape[1]
        generated_txt = tokenizer.decode(outputs[0][input_len:],skip_special_tokens = True)
        res={
            "problem":problem,
            "answer":generated_txt
        }
        f.write(json.dumps(res,ensure_ascii=False)+'\n')
print('inference finish')