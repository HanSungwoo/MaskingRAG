import json
from typing import Tuple,List
import torch
import re
import os
import fire
import random
import math
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
tokenizer = AutoTokenizer.from_pretrained(
    'sentence-transformers/all-mpnet-base-v2',
)

def make_vector_avg_for_each_tag(open_tag_data_path, save_path, dataset_name):
    total_num = [0, 0, 0, 0]  # To count PER, LOC, ORG, MISC
    raw_total_per = []
    raw_total_loc = []
    raw_total_org = []
    raw_total_misc = []
    total_word_list = []
    total_vectors = [[], [], [], []]  # To store vectors for each tag
    total_raw_input_list = []
    
    with open(open_tag_data_path, "r", encoding="utf-8") as f:
        total_raw_input_list = [json.loads(line.strip()) for line in f]
    
    if dataset_name in ["FIN", "conll03"]:
        input_list_name_in_jsonl = "tokens"
    else:
        input_list_name_in_jsonl = "input_list"
    
    # Process each tag
    for total_list in tqdm(total_raw_input_list, desc="Processing data", unit="sample"):
        raw_per, raw_loc, raw_org, raw_misc = [], [], [], []
        for i, tag in enumerate(total_list["tag"]):
            word = total_list[input_list_name_in_jsonl][i]
            if "PER" in tag:
                raw_per.append(word)
                total_num[0] += 1
            elif "LOC" in tag:
                raw_loc.append(word)
                total_num[1] += 1
            elif "ORG" in tag:
                raw_org.append(word)
                total_num[2] += 1
            elif "MISC" in tag:
                raw_misc.append(word)
                total_num[3] += 1
        raw_total_per.append(raw_per)
        raw_total_loc.append(raw_loc)
        raw_total_org.append(raw_org)
        raw_total_misc.append(raw_misc)
    print(total_num)

    tag_names = ["PER", "LOC", "ORG", "MISC"]
    for idx, raw_list in enumerate(
        tqdm([raw_total_per, raw_total_loc, raw_total_org, raw_total_misc], desc="Processing tags", unit="tag")
    ):
        for idx_list, word_list in tqdm(
            enumerate(raw_list), 
            desc=f"Processing {tag_names[idx]} word lists", 
            unit="word list", 
            leave=False
        ):
            sentence = total_raw_input_list[idx_list]["text"]
            word_embedding = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            input_ids = word_embedding["input_ids"]
            with torch.no_grad():
                outputs = model(**word_embedding)
                hidden_states = outputs.last_hidden_state

            total_words = []
            for word in word_list:
                tokens = tokenizer.tokenize(word)
                total_words += list(tokens)
            tokenized_words = tokenizer.convert_ids_to_tokens(input_ids[0])
            for find_word in total_words:
                start_index = 0
                while start_index < len(tokenized_words):
                    try:
                        word_index = tokenized_words.index(find_word, start_index)
                        word_vector = hidden_states[0, word_index, :]
                        total_vectors[idx].append(word_vector)

                        start_index = word_index + 1
                        break
                    except ValueError:
                        print(f"Token '{find_word}' not found starting from index {start_index} in {tokenized_words}")
                        break
    avg_vectors = []
    for vectors in total_vectors:
        if vectors:
            avg_vectors.append(np.mean(vectors, axis=0).tolist())
        else:
            avg_vectors.append([0] * 768)
    
    # Save results
    save_data = {
        "<PER>": avg_vectors[0],
        "<LOC>": avg_vectors[1],
        "<ORG>": avg_vectors[2],
        "<MISC>": avg_vectors[3]
    }
    with open(save_path, "w", encoding="utf-8") as file:
        file.write(json.dumps(save_data, ensure_ascii=False))


def save_vector_each_dataset(dataset_name = None):
    open_tag_data_path = f"MaskingRAG/result/taged_reslut/{dataset_name}/{dataset_name}_train_tag_result.jsonl"
    save_path = f"MaskingRAG/result/taged_reslut/{dataset_name}/{dataset_name}_tag_vector.jsonl"
    make_vector_avg_for_each_tag(open_tag_data_path, save_path, dataset_name)

fire.Fire(save_vector_each_dataset)