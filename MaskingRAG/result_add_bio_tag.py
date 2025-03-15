import json
import numpy as np
import fire
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import LongformerTokenizer, LongformerModel


# train data path
def result_add_bio_tag(
        data_type = None, #google
        ):
    input_file_path = f"MaskingRAG/result/taged_reslut/{data_type}/{data_type}_train.txt"
    raw_input_file_path = f"MaskingRAG/dataset/{data_type}/{data_type}_train.jsonl"
    save_path = f"MaskingRAG/result/taged_reslut/{data_type}/{data_type}_train_tag_result.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(
        "dslim/bert-large-NER",
        )

    input_dict_list = []
    with open(input_file_path, "r", encoding='utf-8') as file:
        for line in file:
            line = eval(line.strip())
            input_dict_list.append(line)

    with open(raw_input_file_path, "r", encoding="utf-8") as f:
        total_raw_input_list = [json.loads(line.strip()) for line in f]
    total_raw_input_list = total_raw_input_list[:len(input_dict_list)]


    total_word_dict_list = []
    for info_line_total in input_dict_list:
        word_dict_list = []
        for info_ner in info_line_total:
            if info_ner["score"] >= 0.80:
                word_dict_list.append(
                    {
                        "word": info_ner["word"],
                        "entity": info_ner["entity"],
                    }
                )
        total_word_dict_list.append(word_dict_list)


    raw_input_to_list = []
    inital_tag_for_raw_input = []
    summary_text_list = []
    remove_index = []
    ans_for_ecqa = []
    explanation_list = []
    i = 0
    for sent in total_raw_input_list:
        input_ids = tokenizer(sent["text"])["input_ids"]
        if len(tokenizer(sent["text"])["input_ids"])> 512:
            print(i, ":", len(tokenizer(sent["text"])["input_ids"]))
            remove_index.append(i)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        summary_text_list.append(sent["summaries"])
        raw_input_to_list.append(tokens[1:-1])
        inital_tag_for_raw_input.append(["O"] * (len(tokens[1:-1])))
        i+=1

    # flag = False
    # for list_a in raw_input_to_list:
    #     if "[SEP]" in list_a or "[CLS]" in list_a:
    #         flag = True
    # print(flag)

    for i, line_list in enumerate(total_word_dict_list):
        for j, word in enumerate(raw_input_to_list[i]):
            for word_dict in line_list:
                if word == word_dict["word"]:
                    inital_tag_for_raw_input[i][j] = word_dict["entity"]
                    break


    final_raw_input_to_list = []
    final_inital_tag_for_raw_input = []

    for j, token_list in enumerate(raw_input_to_list):
        words = []
        new_tags = []
        current_word = ""
        current_tag = inital_tag_for_raw_input[j][0]  # 첫 번째 태그로 초기화
        
        for i, token in enumerate(token_list):
            if token.startswith("##"):
                # '##'을 제거하고 뒤에 이어붙임
                current_word += token[2:]
                # 태그가 'O'가 아니고, 현재와 이전 태그의 엔티티 종류가 다르면 check_same_tag를 False로 설정
                if current_tag != "O" and inital_tag_for_raw_input[j][i] != "O":
                    if current_tag[2:] != inital_tag_for_raw_input[j][i][2:]:
                        check_same_tag = False
                else:
                    check_same_tag = False
            else:
                # current_word가 비어있지 않다면 새로운 단어가 생성된 것이므로 현재까지의 단어와 태그를 추가
                if current_word:
                    words.append(current_word)
                    # check_same_tag가 True일 경우 현재 태그를 유지하고, 그렇지 않으면 "O" 태그 추가
                    new_tags.append(current_tag if check_same_tag else "O")
                
                # 새로운 단어 및 태그 설정
                current_word = token
                current_tag = inital_tag_for_raw_input[j][i]  # 현재 태그 업데이트
                check_same_tag = True  # 새로운 단어 시작 시 check_same_tag 초기화

        # 마지막 단어와 태그 추가
        if current_word:
            words.append(current_word)
            new_tags.append(current_tag if check_same_tag else "O")
        
        final_raw_input_to_list.append(words)
        final_inital_tag_for_raw_input.append(new_tags)


    for i in range(len(final_inital_tag_for_raw_input)):
        if "I-" in final_inital_tag_for_raw_input[i][0]:
            final_inital_tag_for_raw_input[i][0] = "B-" + final_inital_tag_for_raw_input[i][0][2:]
        for j in range(len(final_inital_tag_for_raw_input[i])-1):
            if final_inital_tag_for_raw_input[i][j+1] != "O":
                if final_inital_tag_for_raw_input[i][j] != "O":
                    if final_inital_tag_for_raw_input[i][j][2:] == final_inital_tag_for_raw_input[i][j+1][2:]:
                        if "B-" in final_inital_tag_for_raw_input[i][j+1]:
                            final_inital_tag_for_raw_input[i][j+1] = "I-" + final_inital_tag_for_raw_input[i][j+1][2:]
                    elif final_inital_tag_for_raw_input[i][j][2:] != final_inital_tag_for_raw_input[i][j+1][2:]:
                        if "I-" in final_inital_tag_for_raw_input[i][j+1]:
                            final_inital_tag_for_raw_input[i][j+1] = "B-" + final_inital_tag_for_raw_input[i][j+1][2:]
                elif final_inital_tag_for_raw_input[i][j] == "O":
                    if "I-" in final_inital_tag_for_raw_input[i][j+1]:
                        final_inital_tag_for_raw_input[i][j+1] = "B-" + final_inital_tag_for_raw_input[i][j+1][2:]


    with open(save_path, "w", encoding="utf-8") as file:
        for i, line_list in enumerate(final_raw_input_to_list):
            if i in remove_index: # 만약에 tokenizer가 512이상으로 넘어가면 제외시킨다.
                continue
            save_dict = {
                "tag" : final_inital_tag_for_raw_input[i],
                "input_list" : line_list,
                "text" : total_raw_input_list[i]["text"],
                "summaries" : summary_text_list[i][0]
            }
        
            json.dump(save_dict, file, ensure_ascii=False)
            file.write("\n")

    with open(f"MaskingRAG/result/taged_reslut/{data_type}/remove_idx.jsonl", "w", encoding='utf-8') as file:
        save_dict = {"idx": remove_index}
        json.dump(save_dict, file, ensure_ascii=False)
        file.write("\n")

fire.Fire(result_add_bio_tag)