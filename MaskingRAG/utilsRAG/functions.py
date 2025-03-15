import json
from typing import Tuple,List
from tqdm import tqdm
import torch
import yaml,fire,json,time

def get_dataset(data:str='google') -> Tuple[List,List]:
    path = f"MaskingRAG/dataset/{data}/{data}.jsonl"
    texts,summaries = [],[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            texts.append(line['text'])
            summaries.append(line['summaries'][0])
    return texts,summaries

def get_dataset_ner(data: str = 'conll03', template:str = "null") -> List[str]:
    if data in ["google"]:
        path = f"MaskingRAG/result/taged_reslut/{data}/{data}_test_tag_result.jsonl"
    else:
        path = f"MaskingRAG/dataset/{data}/{data}_test.jsonl"
    token = []
    ner_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            if data == "google":
                token.append(line["input_list"])
                ner_list.append(line["tag"])
            else:
                token.append(", ".join(line['tokens']))
                ner_list.append(line['ner_tags'])
    return token, ner_list

def get_dataset_ner_for_eval(data: str = 'conll03', template:str = "null") -> List[str]:
    if data in ["google"]:
        path = f"result/taged_reslut/{data}/{data}_test_tag_result.jsonl"
    else:
        path = f"dataset/{data}/{data}_test.jsonl"
    token = []
    ner_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            if data == "google":
                token.append(line["input_list"])
                ner_list.append(line["tag"])
            else:
                token.append(line['tokens'])
                ner_list.append(line['ner_tags'])
        # print(token)
    return token, ner_list


def process_batch(batch,tokenizer,generate_kwargs,model):
    model_inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
    gen_out = model.generate(**model_inputs, **generate_kwargs)
    gen_text = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    return gen_text

def batch_loop(data, batch_size, tokenizer, generate_kwargs, model):
    output = []
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i + batch_size]
            gen_text = process_batch(batch,tokenizer,generate_kwargs,model)
            for item in gen_text:
                item = item.replace("$}}%","")
                item = item.replace("\n"," [SEP] ")
                output.append(item)
    return output

def post_processing(data_field:str, outputs:List[str], prompts:List[str], template_type:str, dataset_check:str) -> List[str]:
    post_processed_outputs = []
    for output,prompt in zip(outputs,prompts):
        if data_field == "summary":
            check_point = output.rfind("The sentence without the less important words would be:")
            post_processed_outputs.append(output[check_point+len("The sentence without the less important words would be:"):])
        else:
            prompt = prompt.replace("\n"," [SEP] ")
            post_processed_outputs.append(output.replace(prompt,""))
    return post_processed_outputs
