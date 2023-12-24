import transformers, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from datasets import load_dataset,load_from_disk
from rouge import FilesRouge, Rouge
from prompt_functions import generate_prompt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os, argparse, time


parser = argparse.ArgumentParser()
parser.add_argument('--model', default="Mistral-7B-Instruct-v0.1", type=str, help='LLaMA model')
parser.add_argument('--prune_ratio', default=0, type=float, help='LLaMA model')

parser.add_argument('--data', default="summeval", type=str, help='select a summarization dataset', 
                    # choices=[ #"cogensumm", "frank", 
                    #          "polytope", "factcc", "summeval", "xsumfaith", "rct_summaries", "legal_contracts",
                    #          ]
                             )
parser.add_argument('--prune_method', default="fullmodel", type=str, help='if using pruned model and which to use', 
                    choices=["fullmodel", "wanda", "sparsegpt", "magnitude"])
parser.add_argument('--prompt_id', default="A", type=str, 
                    choices=["A", "B", "C"],
                    help='pick a prompt template from prompt list, A or B or None')

parser.add_argument('--test_num', default=2, type=int)
#parser.add_argument('--cache_model_dir', default="pruned_model", type=str)
args = parser.parse_args()

if args.prune_ratio == 0: assert args.prune_method == "fullmodel"

def get_sequence_length(config: PretrainedConfig, default: int = 2048) -> int:
    if hasattr(config, "sliding_window"):
        return config.sliding_window
    elif hasattr(config, "max_position_embeddings"):
        return config.max_position_embeddings
    else:
        return default



f = open(f"data/{args.data}.json")
dataset = json.load(f)



def get_attention_to_source_list(model, tokenizer, input_ids):


    sequence_length = get_sequence_length(model.config)
    input_len = input_ids.size()[-1]
    output = model.generate(input_ids, max_length=sequence_length, # num_return_sequences=1,
                            output_scores=True, output_attentions=True, output_hidden_states=True,
                            return_dict_in_generate = True) 

    output_text_len = len(output.sequences[-1])
    new_token_len = output_text_len - input_len

    attention = output['attentions']
    tokens_attention_to_source_list = []
    for token_idx in range(1, new_token_len):
        attention_to_source_sum = 0
        for head_idx in range(32):
            attention_to_source = torch.sum(attention[token_idx][31][-1][head_idx][0][:input_len])
            attention_to_source_sum += attention_to_source
        mean_attention_to_source = attention_to_source_sum / 32
        tokens_attention_to_source_list.append(mean_attention_to_source.item())
    return tokens_attention_to_source_list



def get_full_data_attention(model_name, test_num, prompt_id, prune_method):
    if prune_method == "fullmodel":  
        # if "Mistral" in model_name: model_name = "mistralai/"+model_name
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                    cache_dir="llm_weights", 
                                                    trust_remote_code=True, 
                                                    device_map="auto",
                                                    torch_dtype="auto" if torch.cuda.is_available() else None,
                                                    use_auth_token=True,
                                                    ) # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="pruned_model", use_fast=False, trust_remote_code=True)
    

    else: 
        model = AutoModelForCausalLM.from_pretrained(f'pruned_model/{model_name}_{prune_ratio}/{prune_method}',
                                                    # cache_dir="pruned_model", 
                                                    trust_remote_code=True, 
                                                    device_map="auto",
                                                    torch_dtype="auto" if torch.cuda.is_available() else None,
                                                    ) # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
        tokenizer = AutoTokenizer.from_pretrained(f'pruned_model/{model_name}_{prune_ratio}/{prune_method}',
                                                    # cache_dir="pruned_model",  
                                                    use_fast=False, trust_remote_code=True)
    model.eval()
    

    list_of_list = []
    
    for i, key in enumerate(dataset.keys()):
        
        if i <= test_num:
            print(i)
            text = generate_prompt(
                task="summarization",
                model= 'llama',
                prompt_id=prompt_id,
                document=dataset[key]["document"],
                )
            
            input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    

            sequence_length = get_sequence_length(model.config)
            if input_ids.shape[1] > sequence_length:
                print(f"Skipping {key} ({input_ids.shape[1]} > {sequence_length}).")
                continue
            
            lis = get_attention_to_source_list(model, tokenizer, input_ids)
            list_of_list.append(lis)
        else: pass

    return list_of_list




model_family = str(args.model).split("-")[0]  # Llama -2-13b-chat-hf_0.4   falcon -7b-instruct_0.4


if "llama" in str(args.model).lower():
    size = str(args.model).split("-")[2]
    print(' it is a llama size of ', size)
else:
    size = str(args.model).split("-")[1]   # falcon-7b-instruct_0.4

model_shortname = f"{model_family}-{size}"  # falcon-7b
print(' model_shortname: ', model_shortname)



pkl_file_dir = f'./saved_attention/{args.data}/{model_shortname}/{args.prune_method}/Prompt{args.prompt_id}/'
img_file_dir = f'./images/{args.data}/{model_shortname}/{args.prune_method}/Prompt{args.prompt_id}/'

if not os.path.exists(pkl_file_dir): os.makedirs(pkl_file_dir)
if not os.path.exists(img_file_dir): os.makedirs(img_file_dir)



# for full model
if os.path.exists(pkl_file_dir + f'{model_shortname}_full.pkl'): 
    open_file = open(pkl_file_dir + f'{model_shortname}_full.pkl', "rb")
    noprune_model_attention = pickle.load(open_file)
    open_file.close()
else:
    noprune_model_attention = get_full_data_attention(model_name= args.model,
                                                      test_num=args.test_num, 
                                                      prompt_id=args.prompt_id, 
                                                      prune_method=args.prune_method
                                                      )
    open_file = open(pkl_file_dir + f'{model_shortname}_full.pkl', 'wb')
    pickle.dump(noprune_model_attention,open_file)
    open_file.close()


prune_ratio = str(args.model).split("_")[-1]
pruned_model_attention = get_full_data_attention(f"./pruned_model/{args.model}/{args.prune_method}/", 
                                                    args.test_num, args.prompt_id)

open_file = open(pkl_file_dir + f'{model_shortname}_{prune_ratio}.pkl', 'wb')
pickle.dump(pruned_model_attention,open_file)
open_file.close()






def get_mean_std(list_of_list):
    min_length = min(len(sub_list) for sub_list in list_of_list)
    cuted_pruned_attention_list = [sub_list[:min_length] for sub_list in list_of_list]    
    mean = np.mean(cuted_pruned_attention_list, axis=0)
    std = np.std(cuted_pruned_attention_list, axis=0)
    index = np.arange(1, len(mean) + 1).tolist()

    return index, mean, std



index, mean, std = get_mean_std(pruned_model_attention)
plt.plot(index, mean, "red", label=f'{str(args.prune_method).capitalize()}({str(prune_ratio)})')
plt.fill_between(index, mean - std, mean + std, alpha=0.2, color="red")

index, mean, std = get_mean_std(noprune_model_attention)
plt.plot(index, mean, "grey", label="No Pruning")
plt.fill_between(index, mean - std, mean + std, alpha=0.2, color="grey")

plt.legend()
plt.xlabel("Generated new token index")
plt.ylabel("Attention ratio (sum) to source input")
plt.title(f'Attention distribution of {model_shortname} ({prune_ratio})', fontsize=16)
plt.savefig(img_file_dir + f'Prompt{args.prompt_id}_Prune{prune_ratio}.png')


