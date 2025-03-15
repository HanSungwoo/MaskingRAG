from typing import Tuple,List


def get_ner_template(inst_type:str) -> Tuple[str, bool]:
    # print(inst_type)
    if inst_type == "summary_llama3":
        Systems = ''''''
        instruction = '''<|start_header_id|>user<|end_header_id|>
Sentence:\n{input_text}\nThe sentence without the less important words would be:<|eot_id|>
<|start_header_id|>assistant<|end_header_id|> '''

    elif inst_type == "summary_Phi":
        Systems = ''''''
        instruction = '''<|user|>
Sentence:\n{input_text}\nThe sentence without the less important words would be:<|end|>
<|assistant|> '''

    return instruction, Systems