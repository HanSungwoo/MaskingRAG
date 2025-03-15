# MaskingRAG

Example for google summarization dataset.

## RAG Dataset making
First, we perform NER tagging on the training data of each task’s dataset using an NER extraction model.

Next, we map the matched NER tags to the tokens they correspond to.

Third, each mapped token is replaced with a special token that represents its corresponding label, inserted according to a specified ratio. For example, if the rate is set to 1 and the sample sentence is "I am Kim Yun-gyeong", it will be transformed into "I am <PER>."

Finally, to generate the vector for the special token, we calculate the vectors for all the mapped tokens in the entire training dataset, take their average, and designate the result as the special token’s vector.

python3 run code

    python3 MaskingRAG/bert_base_ner_tagging_model.py --data_name google
    python3 MaskingRAG/result_add_bio_tag.py --data_type google
    python3 MaskingRAG/dataset_making_with_make_for_summary.py --dataset_name google
    python3 MaskingRAG/save_vector_each_dataset_original.py --dataset_name google
    
bash file run code

    bash make_mask_dataset.sh


## Rag inference step
python3 run code(llama3)

        CUDA_VISIBLE_DEVICES=0 python3 MaskingRAG/RAG.py --dataset_check google --check_mask_item all --mask_check True --model_type Phi --model_size 8 --masking_rate 0.1

python3 run code(llama3)

        CUDA_VISIBLE_DEVICES=0 python3 MaskingRAG/RAG.py --dataset_check google --check_mask_item all --mask_check True --model_type Phi --model_size 8 --masking_rate 0.1

bash file run code

        bash make_mask_dataset.sh
