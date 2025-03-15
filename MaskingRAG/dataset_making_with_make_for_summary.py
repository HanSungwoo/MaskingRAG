import json
from typing import Tuple,List
import re
import os
import fire
import random
import math
# from tqdm import tqdm
# from transformers import AutoTokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-mpnet-base-v2'
        )
def dataset_make_mask_for_other_set(
        dataset_name = None  #google
        ):
    """
    This function reads a .jsonl file containing NER-tagged data, selectively masks specific entity types
    based on a given 'masking_rate' and 'check_tag' criteria, and writes out the masked results.

    Args:
        dataset_name (str): 
            - The name of the dataset directory containing the .jsonl file with tagging information.
            - Example: "google".
            - The file is expected at 'MaskingRAG/result/taged_reslut/{dataset_name}/{dataset_name}_train_tag_result.jsonl'.

    Returns:
        None:
            - The masked data is written to new .jsonl files in the 
              'MaskingRAG/result/taged_reslut/{dataset_name}/{masking_rate}/...' directory.
    """
    check_tag_list = ["all", "PER-LOC-ORG"]
    masking_rate_list = [0.1, 0.15, 0.3, 0.5, 0.7, 1]
    if dataset_name == "google":
        data_to_put_mask_dataset_path = f"MaskingRAG/result/taged_reslut/{dataset_name}/{dataset_name}_train_tag_result.jsonl"
    else:
        data_to_put_mask_dataset_path = f"MaskingRAG/result/taged_reslut/{dataset_name}/{dataset_name}_test_tag_result.jsonl"


    for check_tag in check_tag_list:
        for masking_rate in masking_rate_list:
            if check_tag == None:
                output_path_directory = f"MaskingRAG/result/taged_reslut/{dataset_name}/{masking_rate}"
                output_path = f"MaskingRAG/result/taged_reslut/{dataset_name}/{masking_rate}/{dataset_name}_tag_result_with_mask_{masking_rate}.jsonl"
                target_elements = ["B-PER", "B-LOC", "B-ORG", "B-MISC"]
                mask_item = tokenizer.mask_token
                replace_word_v1 = "<mask> <mask>"
                replace_word_v2 = "<mask><mask>"

            elif check_tag == "all":
                output_path_directory = f"MaskingRAG/result/taged_reslut/{dataset_name}/{masking_rate}/all/"
                output_path = f"MaskingRAG/result/taged_reslut/{dataset_name}/{masking_rate}/all/{dataset_name}_tag_result_with_mask_{masking_rate}.jsonl"
                target_elements = ["B-PER", "B-LOC", "B-ORG", "B-MISC"]

            elif check_tag == "PER-LOC-ORG":
                output_path_directory = f"MaskingRAG/result/taged_reslut/{dataset_name}/{masking_rate}/per_loc_org/"
                output_path = f"MaskingRAG/result/taged_reslut/{dataset_name}/{masking_rate}/per_loc_org/{dataset_name}_tag_result_with_mask_{masking_rate}.jsonl"
                target_elements = ["B-PER", "B-LOC", "B-ORG"]

            try:
                if not os.path.exists(output_path_directory):
                    os.makedirs(output_path_directory)
            except OSError:
                print("Error: Failed to create the directory.")

            total_text_include_Mask = []


            with open(data_to_put_mask_dataset_path, "r", encoding= "utf-8") as file:
                total_raw_input_list = [json.loads(line.strip()) for line in file]


            mask_total_raw_input_list = []

            for _, line in enumerate(total_raw_input_list):
                tag = line["tag"]
                input_list = line["input_list"]
                text = line["text"]
                raw_text = line["text"]
                summaries = line["summaries"]

                filtered_indices = [i for i, x in enumerate(tag) if x in target_elements]
                sample_size = max(0, round(len(filtered_indices) * masking_rate))

                if len(filtered_indices) == 0:
                    text = text
                    pass
                else:
                    random_indices = random.sample(filtered_indices, sample_size)

                    indices_to_replace = set()
                    for i in random_indices:
                        if check_tag == "all" or check_tag == "PER-LOC-ORG":
                            indices_to_replace.add((i, tag[i][2:]))  
                        else:
                            indices_to_replace.add(i)  

                        j = i + 1
                        compare_tag = "I-" + tag[i][2:]
                        while j < len(tag) and tag[j] == compare_tag and tag[j] != "O":
                            if check_tag == "all" or check_tag == "PER-LOC-ORG":
                                indices_to_replace.add((j, tag[i][2:]))
                            else: 
                                indices_to_replace.add(j)
                            j += 1

                    words = input_list.copy()
                   
                    if check_tag == "all":
                        for i, tagn in indices_to_replace:
                            if i < len(words): 
                                if tagn == "PER":
                                    words[i] = "<PER>"
                                elif tagn == "LOC":
                                    words[i] = "<LOC>"
                                elif tagn == "ORG":
                                    words[i] = "<ORG>"
                                elif tagn == "MISC":
                                    words[i] = "<MISC>"

                    elif check_tag == "PER-LOC-ORG":
                        for i, tagn in indices_to_replace:
                            if i < len(words): 
                                if tagn == "PER":
                                    words[i] = "<PER>"
                                elif tagn == "LOC":
                                    words[i] = "<LOC>"
                                elif tagn == "ORG":
                                    words[i] = "<ORG>"
                    else:
                        for i in indices_to_replace:
                            if i < len(words): 
                                words[i] = mask_item

                    text = " ".join(words)

                    text = text.replace("' s", "'s")
                    text = text.replace("<ORG> . <ORG> .", "<ORG>")
                    text = text.replace("n ' t", "n't")
                    text = text.replace("' re", "'re")
                    while "  " in text:
                        text.replace("  ", " ")
                    if check_tag == "all":
                        replace_word_list = ["<PER>", "<LOC>", "<ORG>", "<MISC>"]
                        for replace_word in replace_word_list:
                            while f"{replace_word} {replace_word}" in text:
                                text = text.replace(f"{replace_word} {replace_word}", replace_word)
                            while f"{replace_word}{replace_word}" in text:
                                text = text.replace(f"{replace_word}{replace_word}", replace_word)
                    elif check_tag == "PER-LOC-ORG":
                        replace_word_list = ["<PER>", "<LOC>", "<ORG>"]
                        for replace_word in replace_word_list:
                            while f"{replace_word} {replace_word}" in text:
                                text = text.replace(f"{replace_word} {replace_word}", replace_word)
                            while f"{replace_word}{replace_word}" in text:
                                text = text.replace(f"{replace_word}{replace_word}", replace_word)
                    else:
                        while replace_word_v1 in text:
                            text = text.replace(replace_word_v1, mask_item)
                        while replace_word_v2 in text:
                            text = text.replace(replace_word_v2, mask_item)
                    
                
                mask_total_raw_input_list.append(text)

                total_text_include_Mask.append({
                    "tag": tag,
                    "input_list": input_list,
                    "text": text,
                    "raw_text" : raw_text,
                    "summaries" : summaries
                })

            # print(type(mask_total_raw_input_list[1]))
            
            # output_path = f"result/taged_reslut/{dataset_name}/{masking_rate}/{check_tag.lower()}/{dataset_name}_tag_result_with_mask_{masking_rate}_test.jsonl"
            with open(output_path, "w", encoding="utf-8") as output_file:
                for line in total_text_include_Mask:
                    json.dump(line, output_file, ensure_ascii=False)
                    output_file.write("\n")
            
            output_path_directory = None
            output_path = None
            target_elements = None
            mask_item = None
            replace_word_v1 = None
            replace_word_v2 = None

fire.Fire(dataset_make_mask_for_other_set)