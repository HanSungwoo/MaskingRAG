import json
import os
import fire
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


#dataset/coin_flip/coin_flip_train.jsonl
def inference(
                data_name = None #google
                ):
    """
    This function takes a dataset name (string) as input, locates and loads the corresponding
    .jsonl file, performs Named Entity Recognition (NER) using a pre-trained model, and then
    writes the results to a text file.

    Args:
        data_name (str): 
            - The name of the dataset (e.g., "google", "esnli", "ecqa", "coin_flip").
            - The function expects a .jsonl file in the 'dataset/<data_name>' directory 
              (e.g., 'dataset/coin_flip/coin_flip_train.jsonl').

    Returns:
        None:
            - The NER results are saved to a .txt file within 'result/taged_reslut/<data_name>/'
              for each set of processed input lines.
    """
        
    dataset_name = data_name + "_train"
        # MaskingRAG/dataset/google/google_train.jsonl
    path = f"MaskingRAG/dataset/{data_name}/{dataset_name}.jsonl"
    input_list = []
    input_list2 = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = json.loads(line)
            input_list.append(line["text"])

    tokenizer = AutoTokenizer.from_pretrained(
        "dslim/bert-large-NER"
        )

    model = AutoModelForTokenClassification.from_pretrained(
        "dslim/bert-large-NER"
        )
    nlp = pipeline(
        "ner",
        model=model, 
        tokenizer=tokenizer,
        # device= "cuda",
        torch_dtype="auto"
        )

    output_list = []
    for i in tqdm(range(0, len(input_list))):
        output_list.append(nlp(input_list[i]))

    result_path = f"MaskingRAG/result/taged_reslut/{data_name}"
    try:
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    except OSError:
        print("Error: Failed to create the directory.")

    result_path += f"/{dataset_name}.txt"

    with open(result_path, "w", encoding = "utf-8") as file:
        for output in output_list:
            file.write(str(output) + "\n")

    if len(input_list2) != 0:
        output_list2 = []
        for i in tqdm(range(0, len(input_list2))):
            output_list2.append(nlp(str(input_list2[i])))

        result_path = f"MaskingRAG/result/taged_reslut/{data_name}"
        try:
            if not os.path.exists(result_path):
                os.makedirs(result_path)
        except OSError:
            print("Error: Failed to create the directory.")

        result_path += f"/{dataset_name}_sentence_2.txt"

        with open(result_path, "w", encoding = "utf-8") as file:
            for output in output_list2:
                file.write(str(output) + "\n")

fire.Fire(inference)