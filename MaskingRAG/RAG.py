from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from transformers import BitsAndBytesConfig, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import fire
import faiss
import json
import numpy as np
import os
import torch
from utilsRAG import (
    batch_loop,
    post_processing,
    get_ner_template,
)

# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def load_data(file_path: str):
    """
    Load data from a .jsonl file.
    Args:
        file_path (str): - Path to the JSON lines file.
    Returns:
        list: - A list of dictionaries loaded from the file.
    """
    with open(file_path, 'r', encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    return data

def create_vector_store(data_field: str, data: list, model_for_sentence, dataset_check: str) -> FAISS:
    """
    Create a FAISS vector store from the provided data.

    Args:
        data_field (str): The field name to reference in 'data' (although in this function, it mostly prints).
        data (list): A list of data entries (each is typically a dictionary containing 'text' and possibly other fields).
        model_for_sentence (SentenceTransformer): A SentenceTransformer model used to encode the text into vectors.
        dataset_check (str): Name or identifier of the dataset (for logging or debugging purposes).

    Returns:
        FAISS:A FAISS vector store (LangChain Community's FAISS wrapper) that contains the indexed vectors and references to the original documents.
    """
    print(data_field)
    texts = [entry['text'].strip() for entry in data]

    vectors = np.array([model_for_sentence.encode(text) for text in tqdm(texts, desc="Encoding texts")], dtype=np.float32)
    
    faiss_index = faiss.IndexFlatL2(vectors.shape[1])
    faiss_index.add(vectors)
    
    docstore = {i: entry for i, entry in enumerate(data)}
    index_to_docstore_id = lambda idx: str(idx)
    
    vector_store = FAISS(
        index=faiss_index,
        embedding_function=model_for_sentence.encode,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    
    return vector_store

def find_most_similar(vector_store: FAISS, query_vector: np.ndarray, k: int = 5):
    """
    Query the FAISS index for the k most similar items to a given vector.

    Args:
        vector_store (FAISS):  The FAISS vector store containing the index.
        query_vector (np.ndarray):  The query vector in float32 format.
        k (int, optional):  Number of nearest neighbors to return. Defaults to 5.

    Returns:
        distances (np.ndarray): Array of distances for the nearest neighbors. 
        indices (np.ndarray): Array of FAISS indices for the nearest neighbors.
    """
    query_vector = np.array(query_vector, dtype=np.float32)
    distances, indices = vector_store.index.search(np.array([query_vector]), k)
    return distances[0], indices[0]

def text_to_vector(text, model_for_sentence):
    """
    Convert a text string to a vector using a SentenceTransformer model.

    Args:
        text (str): The input text to encode.
        model_for_sentence (SentenceTransformer): The SentenceTransformer model used for encoding.

    Returns:
            Numpy array of float32 representing the encoded text.
    """
    return model_for_sentence.encode(text).astype(np.float32)

def create_few_shot_prompt_set(
    mask_check: bool, 
    data_field: str, 
    input_text: str, 
    vector_store: FAISS, 
    model, 
    template_type: str, 
    num_examples: int, 
    model_type: str, 
    dataset_check: str
):
    """
    Create a few-shot prompt for RAG-style inference.

    This function fetches the nearest neighbors from the vector store to serve as "few-shot" examples,
    then formats them using a template retrieved from `get_ner_template`.

    Args:
        mask_check (bool):  Whether the data is masked or not.
        data_field (str):  Field name (unused in the main logic here, but kept for consistency).
        input_text (str):  The input text query for which we want to find examples.
        vector_store (FAISS):  The FAISS vector store containing the data.
        model (SentenceTransformer):  The model used to encode the query text into a vector.
        template_type (str):  Template identifier used by `get_ner_template` to format prompts.
        num_examples (int):  Number of examples to retrieve from the vector store.
        model_type (str):  Type of the model ("llama3", "Phi", etc.) for any special formatting in the template.
        dataset_check (str):  Identifier for the dataset (logging or use in templates).

    Returns:
        tuple: A list of example strings formatted according to the template, System.
    """

    query_vector = text_to_vector(input_text, model)
    distances, indices = find_most_similar(vector_store, query_vector, k=num_examples)
    # print(indices)
    examples = []
    template, System = get_ner_template(template_type)
    for idx in indices:
        example = vector_store.docstore[idx]
        answer_finall = ''
        answer = []
        if mask_check:  
            text_list = example['raw_text']
        else:
            text_list = example["text"]
        answer.append(example['summaries'])
        if model_type == "llama3":
            answer_finall = template.format(input_text=text_list) + str(answer[0]) + " <|eot_id|>\n"
        elif model_type == "Phi" or model_type == "phi":
            answer_finall = template.format(input_text=text_list) + str(answer[0]) + " <|end|>\n"
        examples.append(answer_finall)

    return examples, System

def get_input_test_for_prompt(
    data_field: str, 
    test_data_file: str, 
    template_type: str, 
    dataset_check: str, 
    model_type: str
    ):
    """
    Retrieve test prompts from a JSON lines file, applying a template to each entry.

    Args:
        data_field (str):  Field name (unused in the logic, but provided for consistency).
        test_data_file (str):  Path to the test .jsonl file.
        template_type (str):  Template identifier to use when formatting.
        dataset_check (str):  Identifier for the dataset.
        model_type (str):  Identifier for the model type ("llama3", "Phi", etc.).

    Returns:
        tuple: A list of formatted prompt strings. A list of the original text fields (e.g., "text") used for RAG retrieval.
    """
    prompt_set = []
    datas = []
    datas_token_only = []
    template, _ = get_ner_template(template_type)
    with open(test_data_file, 'r', encoding="utf-8") as file:
        datas = [json.loads(line) for line in file]
    # datas = datas[:300]
    print(len(datas))
    # print(datas[1]["question"])
    for data in datas:
        datas_token_only.append(data["text"])
        prompt_set.append(template.format(input_text=data["text"]))
                
    return prompt_set, datas_token_only

def model_ouput_save(
    data_field: str, 
    path: str, 
    outputs: list, 
    model_size: str, 
    model_type: str, 
    dataset_check: str, 
    template_type: str, 
    prompts: list, 
    check_mask_item: str
    ):
    """
    Post-process model outputs and save them to files.

    Args:
        data_field (str):  Field name (e.g., "summary").
        path (str):  Base directory path to store the results.
        outputs (list):  Raw outputs from the language model.
        model_size (str):  Size or variant of the model (e.g., "3", "8").
        model_type (str):  Model identifier (e.g., "llama3", "Phi").
        dataset_check (str):  Identifier of the dataset.
        template_type (str):  Template identifier used during inference.
        prompts (list):  Input prompts (used for post-processing).
        check_mask_item (str): Which masks or special tokens are being considered ("all", "per", etc.) or None.

    Returns:
        None: Saves both raw and post-processed outputs to files.
    """
    try:
        if not os.path.exists(path+f"result/"):
            os.makedirs(path+f"result/")
    except OSError:
            print("Error: Failed to create the directory.")
    path = path + "result/"
    try:
        if not os.path.exists(path+f"output/"):
            os.makedirs(path+f"output/")
    except OSError:
            print("Error: Failed to create the directory.")
    try:
        if not os.path.exists(path+f"output/{dataset_check}"):
            os.makedirs(path+f"output/{dataset_check}")
    except OSError:
            print("Error: Failed to create the directory.")
    try:
        if not os.path.exists(path+f"output/{dataset_check}/{template_type}"):
            os.makedirs(path+f"output/{dataset_check}/{template_type}")
    except OSError:
            print("Error: Failed to create the directory.")

    try:
        if not os.path.exists(path+f"post/"):
            os.makedirs(path+f"post/")
    except OSError:
            print("Error: Failed to create the directory.")
    try:
        if not os.path.exists(path+f"post/{dataset_check}"):
            os.makedirs(path+f"post/{dataset_check}")
    except OSError:
            print("Error: Failed to create the directory.")
    try:
        if not os.path.exists(path+f"post/{dataset_check}/{template_type}"):
            os.makedirs(path+f"post/{dataset_check}/{template_type}")
    except OSError:
            print("Error: Failed to create the directory.")

    post_processed_outputs = post_processing(data_field, outputs, prompts, template_type, dataset_check)

    if check_mask_item != None:
        try:
            if not os.path.exists(path+f"post/{dataset_check}/{template_type}/{check_mask_item}"):
                os.makedirs(path+f"post/{dataset_check}/{template_type}/{check_mask_item}")
        except OSError:
                print("Error: Failed to create the directory.")
        try:
            if not os.path.exists(path+f"output/{dataset_check}/{template_type}/{check_mask_item}"):
                os.makedirs(path+f"output/{dataset_check}/{template_type}/{check_mask_item}")
        except OSError:
                print("Error: Failed to create the directory.")
        #Saving code
        with open(path+f"output/{dataset_check}/{template_type}/{check_mask_item}/{model_type}-{model_size}-template:{template_type}.txt",'w', encoding="utf-8") as f:
            f.write(u"\n".join(outputs))
        with open(path+f"post/{dataset_check}/{template_type}/{check_mask_item}/{model_type}-{model_size}-template:{template_type}.txt",'w', encoding="utf-8") as f:
            f.write("\n".join(post_processed_outputs))
           
    else:
        #Saving code
        with open(path+f"output/{dataset_check}/{template_type}/{model_type}-{model_size}-template:{template_type}.txt",'w', encoding="utf-8") as f:
            f.write(u"\n".join(outputs))
        with open(path+f"post/{dataset_check}/{template_type}/{model_type}-{model_size}-template:{template_type}.txt",'w', encoding="utf-8") as f:
            f.write("\n".join(post_processed_outputs))
            
    print(path+f"post/{dataset_check}/{template_type}/{model_type}-{model_size}-template:{template_type}.txt")
    print(post_processed_outputs[:1])

def model_inference(model_type: str, model_size: str, prompts: list, batch_size: int) -> list:
    """
    Load a quantized language model and generate outputs for the given prompts.

    Args:
        model_type (str):  Model identifier (e.g., "llama3" or "Phi").
        model_size (str):  Model size variant (e.g., "3", "8").
        prompts (list):  List of prompt strings for generation.
        batch_size (int):  Number of prompts to batch together during generation.

    Returns:
        list: The generated outputs in sequence for the given prompts.
    """
    if model_type == "llama3":
        if model_size == "8":
            model_id = f"meta-llama/Meta-Llama-3.1-{model_size}B-Instruct"
        # elif model_size == '70':
        #     model_id = f"meta-llama/Llama-2-{model_size}b-hf"
    elif model_type == "Phi" or model_type == "phi":
        model_type = "Phi"
        model_size = 3
        model_id = f"microsoft/Phi-3-mini-128k-instruct"

    print(model_id)

    nf8_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    if model_type == "Phi":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            quantization_config = nf8_config,
            trust_remote_code=True
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            quantization_config = nf8_config
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if model_type == "llama3":
        generate_kwargs = {
            "max_new_tokens": 150, 
            "do_sample": False,
            }
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    elif model_type == "Phi":
        generate_kwargs = {
            "max_new_tokens": 150, 
            "do_sample": False,
            }
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    model.eval()
    outputs = batch_loop(prompts, batch_size, tokenizer, generate_kwargs, model)

    return outputs


def inference(
    model_type: str = "-",      
    template_type: str = "-",   
    model_size: str = "3",      
    batch_size: int = 1,        
    dataset_check: str = "o",   
    mask_check: str = "-",      
    num_examples: int = 5,      
    masking_rate: float = None, 
    using_data_set: str = "-",  
    check_mask_item: str = None,
    ) -> None:
    """
    Main RAG inference function that loads data, builds a FAISS vector store, retrieves few-shot examples,
    generates final outputs, and saves them.

    Args:
        model_type (str, optional): Model identifier for LLM ("llama3" or "Phi"). Default is "-".
        template_type (str, optional): Template type/name for formatting prompts (e.g., "summary_llama3"). Default is "-".
        model_size (str, optional): Model size (e.g., "3", "8"). Default is "3".
        batch_size (int, optional): Number of prompts to process in a single batch during generation. Default is 1.
        dataset_check (str, optional): Identifier for the dataset (e.g., "google"). Default is "o".
        mask_check (str, optional): Flag or indicator for using masked data. Default is "-".
        num_examples (int, optional): Number of retrieved examples for few-shot RAG. Default is 5.
        masking_rate (float, optional): The fraction of tokens to be masked (if mask_check is true). Default is None.
        using_data_set (str, optional): The name of the dataset actually being used for retrieval. Default is "-".
        check_mask_item (str, optional): Specific tag for mask items, such as "all", "per_loc_org", or None. Default is None.

    Returns:
        None: Results (raw and post-processed) are saved into output files in the specified directory structure.
    """
    if dataset_check in "google":
        template_type = "summary"
        using_data_set = dataset_check
    else:
        template_type = dataset_check
        using_data_set = dataset_check

    if template_type == "summary":
        data_field = "summary"
        data_field_path = "mask_summary"

    print(f"""
    model_type:{model_type}
    model_size:{model_size}
    template_type:{template_type}
    dataset:{dataset_check}
    batch_size:{batch_size}
    mask_check:{mask_check}
    num_examples:{num_examples}
    masking_rate:{masking_rate}
    using_data_set:{using_data_set}
    check_mask_item:{check_mask_item}
""")

    model_size = str(model_size)
    template_type = str(template_type)
    path = f"MaskingRAG/result/{data_field_path}/{model_type}/{dataset_check}/{num_examples}/using_data_{using_data_set}/"

    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
            print("Error: Failed to create the directory.")

    if mask_check:
        path = path + f"MASK_ON/{masking_rate}/"
    else:
        masking_rate = None
        path = path + f"MASK_OFF/"
    print(path)
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
            print("Error: Failed to create the directory.")

    if mask_check:
        if check_mask_item == None:
            data_file = f'MaskingRAG/result/taged_reslut/{using_data_set}/{masking_rate}/{using_data_set}_tag_result_with_mask_{masking_rate}.jsonl'
        else:
            data_file = f'MaskingRAG/result/taged_reslut/{using_data_set}/{masking_rate}/{check_mask_item}/{using_data_set}_tag_result_with_mask_{masking_rate}.jsonl'
    else:
        data_file = f'MaskingRAG/result/taged_reslut/{using_data_set}/{using_data_set}_train_tag_result.jsonl'

    print(data_file)
    if template_type == "summary":
        test_data_file = f"MaskingRAG/dataset/{dataset_check}/{dataset_check}_test.jsonl"

    template_type = f"{template_type}_{model_type}"

    data = load_data(data_file)
    print(data[1])
    model_for_sentence = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    if check_mask_item != None:
        tokenizer_for_include_word_vector = model_for_sentence.tokenizer
        vector_path = f"MaskingRAG/result/taged_reslut/{using_data_set}/{using_data_set}_tag_vector.jsonl"
        print("vector_path :", vector_path)
        with open(vector_path, "r", encoding="utf-8") as f:
            datas = [json.loads(line) for line in f]
        vector_tag = list(datas[0].keys())
        vectors_dict = [torch.tensor(v, dtype=torch.float32) for v in datas[0].values()]

        old_embedding_size = len(tokenizer_for_include_word_vector)
        num_added_tokens = tokenizer_for_include_word_vector.add_tokens(vector_tag)
        print(f"Tokenizer size before: {old_embedding_size}, after: {len(tokenizer_for_include_word_vector)}")
        print(f"Added {num_added_tokens} new tokens.")

        transformer_model = model_for_sentence._first_module().auto_model
        transformer_model.resize_token_embeddings(len(tokenizer_for_include_word_vector), mean_resizing=False)
        embedding_layer = transformer_model.get_input_embeddings()

        for i, vector in enumerate(vectors_dict):
            token_id = old_embedding_size + i
            embedding_layer.weight.data[token_id] = vector
    
    vector_store = create_vector_store(data_field, data, model_for_sentence, dataset_check)
    prompts = []
    prompt_set, datas_token_only = get_input_test_for_prompt(data_field, test_data_file, template_type=template_type, dataset_check = dataset_check, model_type = model_type)
    print(prompt_set[1])
    print("-"*80)
    for idx, tokens in enumerate(datas_token_only):
        rag_few_shot_prompt, System_input = create_few_shot_prompt_set(mask_check, data_field, tokens, vector_store, model_for_sentence, template_type, num_examples=num_examples, model_type=model_type, dataset_check = dataset_check)
        prompt = System_input
        # print(prompt)
        for rag_shot_prompt in rag_few_shot_prompt: 
            prompt += str(rag_shot_prompt)
        prompt += str(prompt_set[idx])
        prompts.append(prompt)
    print(prompts[1])
    print("-" * 90)
    outputs = model_inference(model_type, model_size, prompts, batch_size)
    
    if template_type == "summary":
        print(outputs[1][outputs[1].rfind("The sentence without the less important words would be:"):])

    model_ouput_save(data_field, path, outputs, model_size, model_type, dataset_check, template_type, prompts, check_mask_item)

if __name__ == "__main__":
    fire.Fire(inference)