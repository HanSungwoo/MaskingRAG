import json
from transformers import AutoTokenizer, AutoModelForTokenClassification

output_path = "MaskingRAG/dataset/google/google_test.jsonl"
save_path = ""
tokenizer = AutoTokenizer.from_pretrained(
    "dslim/bert-large-NER"
    )


dataset = []
with open(output_path, "r", encoding="utf-8") as f:
    for lines in f:
        line = json.loads(lines)
        dataset.append(line['text'])


raw_input_to_list = []

for sent in dataset:
    input_ids = tokenizer(sent)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    raw_input_to_list.append(tokens[1:-2])

print(raw_input_to_list[1])
print(dataset[1])

