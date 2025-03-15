import fire
import json
from evaluate_utils import(
    evaluate
)
import os
import nltk
# nltk.download('punkt_tab')

def main(
        mode_name = "llama3", #Phi
        mask_check = True,
        data_field = "summary",
        data_name = "google", #google
        example_num = 5,
        check_maks_tag = "all" #all, None, per_loc_org
        ):
    target_summary, src_list = [], []
    eval_result = []
    
    target_path = f"MaskingRAG/dataset/{data_name}/{data_name}_test.jsonl"
    using_data_set = data_name
        
    if mode_name == "phi":
        mode_name = "Phi"

    eval_save_path =f"MaskingRAG/result/mask_{data_field}/{mode_name}/{data_name}/{example_num}/using_data_{using_data_set}/result/{check_maks_tag}"

    try:
        if not os.path.exists(eval_save_path):
            os.makedirs(eval_save_path)
    except OSError:
            print("Error: Failed to create the directory.")

    eval_save_path += "/eval_result.txt"

    with open(target_path, "r", encoding="utf-8") as file:
        total_dataset_list = [json.loads(line.strip()) for line in file]
    for line in total_dataset_list:
        target_summary.append(line["summaries"][0])
        src_list.append(line["text"])

    for mask_check in [True, False]:
        if mask_check:
            mask_state = "MASK_ON"
            masking_rate_list = [0.1, 0.15, 0.3, 0.5, 0.7, 1]
            # masking_rate_list = [0.1]
        else:
            mask_state = "MASK_OFF"
            masking_rate_list = [None]
        for masking_rate in masking_rate_list:
            print(masking_rate)
            print(mask_state)
            if mode_name == "llama3":
                model_size = 8
            elif mode_name == "Phi":
                mode_name = "Phi"
                model_size = 8
            if mask_check:
                if check_maks_tag in ['per_loc_org', 'all']:
                    path = f"MaskingRAG/result/mask_{data_field}/{mode_name}/{data_name}/{example_num}/using_data_{using_data_set}/{mask_state}/{masking_rate}/result/post/{data_name}/{data_field}_{mode_name}/{check_maks_tag}/{mode_name}-{model_size}-template:{data_field}_{mode_name}.txt"
                    save_path = f"MaskingRAG/result/mask_{data_field}/{mode_name}/{data_name}/{example_num}/using_data_{using_data_set}/{mask_state}/{masking_rate}/result/post/{data_name}/{data_field}_{mode_name}/{check_maks_tag}/{mode_name}-{model_size}-template:{data_field}_{mode_name}_V3.txt"
                else:
                    path = f"MaskingRAG/result/mask_{data_field}/{mode_name}/{data_name}/{example_num}/using_data_{using_data_set}/{mask_state}/{masking_rate}/result/post/{data_name}/{data_field}_{mode_name}/{check_maks_tag}/{mode_name}-{model_size}-template:{data_field}_{mode_name}.txt"
                    save_path = f"MaskingRAG/result/mask_{data_field}/{mode_name}/{data_name}/{example_num}/using_data_{using_data_set}/{mask_state}/{masking_rate}/result/post/{data_name}/{data_field}_{mode_name}/{check_maks_tag}/{mode_name}-{model_size}-template:{data_field}_{mode_name}_V3.txt"
            else:
                path = f"MaskingRAG/result/mask_{data_field}/{mode_name}/{data_name}/{example_num}/using_data_{using_data_set}/{mask_state}/result/post/{data_name}/{data_field}_{mode_name}/{mode_name}-{model_size}-template:{data_field}_{mode_name}.txt"
                save_path = f"MaskingRAG/result/mask_{data_field}/{mode_name}/{data_name}/{example_num}/using_data_{using_data_set}/{mask_state}/result/post/{data_name}/{data_field}_{mode_name}/{mode_name}-{model_size}-template:{data_field}_{mode_name}_V3.txt"

            output_dataset = []
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    output_dataset.append((line.split("  [SEP]  [SEP] "))[0].strip())
                # output_dataset = [line.strip() for line in file]

            process_output_dataset = []

            for output in output_dataset:
                process_output_dataset.append(output.strip().replace("[SEP] assistant", ""))

            with open(save_path, "w", encoding="utf-8") as file:
                file.write(u"\n".join(process_output_dataset))

            eval_result.append(str(masking_rate) + "\n" +
             mask_state + "\n" 
             + save_path + "\n"
             + evaluate(tgts=target_summary, srcs=src_list, hyps=process_output_dataset)
             + ("-"*100) + "\n" )

    with open(eval_save_path, "w", encoding="utf-8") as file:
        file.write(u"\n".join(eval_result))

fire.Fire(main)