import torch
import torch.utils.data.dataloader
from torch.utils.data.dataset import Dataset
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
import os
import json
from tqdm import tqdm
from lsrl import patch_qwen2_for_multi_gpus,CPUAdamW
import time


class MyDataset(torch.utils.data.Dataset):
    def __init__(self,data_path,tokenizer,max_length = 1024):#
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_and_process_data(data_path)

    def _load_and_process_data(self,data_path):
        res = []
        with open(data_path,'r',encoding='utf-8') as f:
            for line in f:
                item=json.loads(line.strip())
                problem = item['problem']
                solution= item['solution']
                full_text = f'<s>用户:{problem}\n助手:{solution}</s>'
                tokenizer_output = self.tokenizer(
                    full_text,
                    max_length = self.max_length,
                    truncation = True,
                    return_tensors = 'pt',
                )
                res.append({
                    "input_ids":tokenizer_output["input_ids"].squeeze(0),
                    "attention_mask":tokenizer_output["attention_mask"].squeeze(0)
                })
        return res
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

def main():
    model_name = '/home/dell/yzt/qwen2.5-14b-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device_count = torch.cuda.device_count()
    devices = list(range(device_count))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.bfloat16,
        )
    patch_qwen2_for_multi_gpus(model,devices = devices)

    train_dataset = MyDataset(data_path='/home/dell/yzt/train.json',tokenizer=tokenizer)
    model.train()
    model.gradient_checkpointing_enable()
    # 至此，pp部署完毕
    opt = CPUAdamW(model.parameters(),lr = 1e-6,accum_steps=16,weight_decay=0.01,eps = 1e-8,grad_offload=False)


    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
    
    train_batch_size = 2
    num_train_epochs = 2
    logging_steps = 10
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = train_batch_size,
        collate_fn=data_collator,
        shuffle = True,
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    global_step = 0
    total_loss = 0
    max_grad_norm = 1.0
    print("training started")
    for epoch in range(num_train_epochs):
        print(f"epoch:{epoch+1}")
        for step,batch in enumerate(tqdm(train_dataloader,desc=f"Epoch {epoch+1}")):
            global_step += 1
            tic = time.time()
            inputs = batch['input_ids'].to(f"cuda:{devices[0]}")
            attention_mask = batch['attention_mask'].to(f"cuda:{devices[0]}")
            labels = inputs.clone()
            outputs = model(inputs,attention_mask = attention_mask,labels = labels,use_cache=False)
            # loss = outputs.loss/accum_steps
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
            if opt.step():
                opt.zero_grad()
                if global_step%logging_steps == 0:
                    avg_loss = total_loss/logging_steps
                    tqdm.write(f"Step {global_step} (Accumulating) |avg_loss:{avg_loss} |Current Accumulated Loss/Step: {total_loss / logging_steps:.6f} | Time: {time.time()-tic:.2f}s")
                    total_loss = 0.0
            else:
                if global_step % logging_steps == 0:
                    tqdm.write(f"Step {global_step} (Accumulating) | Current Accumulated Loss/Step: {total_loss / logging_steps:.6f} | Time: {time.time()-tic:.2f}s")
                    total_loss = 0.0
    print("Finish")
    output_dir = './res'
    os.makedirs(output_dir,exist_ok = True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
    main()