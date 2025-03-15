# MaskingRAG

Example for google summarization dataset.

## RAG Dataset making
First, we perform NER tagging on the training data of each task’s dataset using an NER extraction model.

Next, we map the matched NER tags to the tokens they correspond to.

Third, each mapped token is replaced with a special token that represents its corresponding label, inserted according to a specified ratio. For example, if the rate is set to 1 and the sample sentence is "I am Kim Yun-gyeong", it will be transformed into "I am <PER>."

Finally, to generate the vector for the special token, we calculate the vectors for all the mapped tokens in the entire training dataset, take their average, and designate the result as the special token’s vector.


- python3 run code(For dataset google)

        python3 MaskingRAG/bert_base_ner_tagging_model.py --data_name google
        python3 MaskingRAG/result_add_bio_tag.py --data_type google
        python3 MaskingRAG/dataset_making_with_make_for_summary.py --dataset_name google
        python3 MaskingRAG/save_vector_each_dataset_original.py --dataset_name google

    
- bash file run code

        bash make_mask_dataset.sh


## Rag inference step
- Data Preparation for RAG Retrieval

    - The dataset is constructed to match the specified masking ratio. (0.1, 0.15, 0.3, 0.5, 0.7, 1)
    - For retrieval, the dataset includes special tokens (e.g., <PER>, <LOC>, <ORG>, <MISC>) that are inserted according to the masking ratio.
    - During similarity computation, the index of the closest matching data point is returned.
- Few-Shot Setup

    - In the few-shot configuration, raw data without any special tokens is used.
    - This ensures that the few-shot examples remain unaltered, allowing the model to operate on natural, unmodified data.
    - This distinction enables us to leverage the structured benefits of special tokens for effective retrieval while maintaining the integrity of raw data for few-shot learning.

- python3 run code(For phi-3-mini-128k(dataset: conll03, batch size: 10, masking ratio: 0.1, few shot num(num_examples): 5))
    - python code with mask(with special tokens)
      
            CUDA_VISIBLE_DEVICES=0 python3 MaskingRAG/RAG.py /
               --dataset_check google /
               --num_examples 5 /
               --check_mask_item all /
               --mask_check True /
               --model_type Phi /
               --model_size 8 /
               --masking_rate 0.1 /
               --batch_size 10

    - python code without mask(without special tokens)
      
           CUDA_VISIBLE_DEVICES=0 python3 MaskingRAG/RAG.py /
              --dataset_check google /
              --num_examples 5 /
              --check_mask_item all /
              --mask_check False /
              --model_type Phi /
              --model_size 8 /
              --masking_rate 0.1 /
              --batch_size 10

- python3 run code(For Llama3-8B-Instruct(dataset: google, batch size: 10, masking ratio: 0.1, few shot num(num_examples): 5)

        CUDA_VISIBLE_DEVICES=0 python3 MaskingRAG/RAG.py /
          --dataset_check google /
          --num_examples 5 /
          --check_mask_item all /
          --mask_check True /
          --model_type llama3 /
          --model_size 8 /
          --masking_rate 0.1 /
          --batch_size 10

- bash file run code(For llama3 and dataset: google, batch size: 10, masking ratio: 0.1, few shot num(num_examples): 5)

        bash make_mask_dataset.sh

Additionally, in check_mask_item, using all means that <PER>, <ORG>, <LOC>, and <MISC> are used, while changing it to per_loc_org uses only <PER>, <ORG>, and <LOC>.

## Eval step
- Inference Completion
    - After completing the inference step, run the evaluation for all masking ratios (0.1, 0.15, 0.3, 0.5, 0.7, 1) as well as for no masking (None).

- Result Saving
    - Once all evaluation results are generated, they are saved under the following directory:

                MaskingRAG/result/mask_summary/llama3/{data_name}/{num_examples}/using_data_{data_name}/result/

    - This structure ensures that all experiment results are organized and easily accessible for further analysis.
 
    - python3 run code(For phi-3-mini-128k)
      
                    python3 MaskingRAG/remove_output_and_eval_summary.py /
                     --mode_name Phi /
                     --check_maks_tag all /
                     --data_name google /
                     --example_num 5

    - python3 run code(For Llama3-8B-Instruct)
      
                    python3 MaskingRAG/remove_output_and_eval_summary.py /
                     --mode_name llama3 /
                     --check_maks_tag all /
                     --data_name google /
                     --example_num 5 
