import json
import random

def replace_text(text:str, ratio:int=0.1):
    word_list = text.split(" ")
    length = len(word_list)
    replace_word_num = round(length*ratio)

    rand_idices = random.sample(range(length), replace_word_num)
    rand_idices = sorted(rand_idices, reverse=True)
    # print(rand_idices)
    
    for i in rand_idices:
        del word_list[i]

    return " ".join(word_list)


def main():
    file_name = ["train"]
    ratio = [0.1, 0.15, 0.3]

    for r in ratio:
        for n in file_name:
            new = open(f"/data/seungwoohan/MaskingRAG/result_for_random/google/google_{n}_{str(r)}.jsonl", 'a')

            with open(f"./google_{n}.jsonl",'r') as f:
                datas = [json.loads(line) for line in f]   # data.keys = 'id', 'text', 'summaries'

            for data in datas:
                temp = {}
                temp['raw_text'] = data['text']
                temp['replace_text'] = replace_text(data['text'], ratio=r)
                temp['label'] = data['summaries']

                new.write(json.dumps(temp) + '\n')


if __name__ == '__main__':
    main()