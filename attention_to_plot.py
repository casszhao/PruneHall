import json
# from datasets import load_dataset,load_from_disk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os, argparse, time

def get_mean_std(list_of_list):
    min_length = min(len(sub_list) for sub_list in list_of_list)
    cuted_pruned_attention_list = [sub_list[:min_length] for sub_list in list_of_list]    
    mean = np.mean(cuted_pruned_attention_list, axis=0)
    std = np.std(cuted_pruned_attention_list, axis=0)
    index = np.arange(1, len(mean) + 1).tolist()
    return index, mean, std

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="Llama-2-13b-chat-hf_0.2", type=str, help='LLaMA model', 
                        # choices=[
                        #          "falcon-7b-instruct",
                        #          "mpt-7b-instruct",
                        #          ]
                    )
parser.add_argument('--data', default="summeval", type=str, help='select a summarization dataset', 
                    # choices=[ #"cogensumm", "frank", 
                    #          "polytope", "factcc", "summeval", "xsumfaith", "rct_summaries", "legal_contracts",
                    #          ]
                             )
parser.add_argument('--prune_method', default="sparsegpt", type=str, help='if using pruned model and which to use', 
                    choices=["fullmodel", "wanda", "sparsegpt", "magnitude"])
parser.add_argument('--prompt_id', default="A", type=str, 
                    choices=["A", "B", "C"],
                    help='pick a prompt template from prompt list, A or B or None')

args = parser.parse_args()



model_family = str(args.model).split("-")[0]  # Llama -2-13b-chat-hf_0.4   falcon -7b-instruct_0.4


if "llama" in str(args.model).lower():
    size = str(args.model).split("-")[-3]
    print(' it is a llama size of ', size)
else:
    size = str(args.model).split("-")[-2]   # falcon-7b-instruct_0.4

model_shortname = f'{model_family}-{size}'  # falcon-7b
print(' model_shortname: ', model_shortname)


pkl_file_dir = f'./saved_attention/{args.data}/{model_shortname}/{args.prune_method}/Prompt{args.prompt_id}/'
img_file_dir = f'./images/{args.data}/{model_shortname}/{args.prune_method}/Prompt{args.prompt_id}/'




prune_ratio="0.2"
colour="orange"
file_path = pkl_file_dir + f'{model_shortname}_{prune_ratio}.pkl'
print(f"==>> file_path: {file_path}")
open_file = open(file_path, 'rb')
pruned_model_attention = pickle.load(open_file)
open_file.close()
index, mean, std = get_mean_std(pruned_model_attention)
plt.plot(index, mean, colour, label=str(args.prune_method).capitalize()+ ' ' + str(prune_ratio))
plt.fill_between(index, mean - std, mean + std, alpha=0.2, color=colour)


prune_ratio="0.5"
colour="yellow"
file_path = pkl_file_dir + f'{model_shortname}_{prune_ratio}.pkl'
print(f"==>> file_path: {file_path}")
open_file = open(file_path, 'rb')
pruned_model_attention = pickle.load(open_file)
open_file.close()
index, mean, std = get_mean_std(pruned_model_attention)
plt.plot(index, mean, colour, label=str(args.prune_method).capitalize()+ ' ' + str(prune_ratio))
plt.fill_between(index, mean - std, mean + std, alpha=0.2, color=colour)


# for full model
open_file = open(pkl_file_dir + f'{model_shortname}_full.pkl', "rb")
noprune_model_attention = pickle.load(open_file)
open_file.close()
index, mean, std = get_mean_std(noprune_model_attention)
plt.plot(index, mean, "grey", label="No Pruning")
plt.fill_between(index, mean - std, mean + std, alpha=0.2, color="grey")






plt.legend()
plt.xlabel("Generated new token index")
plt.ylabel("Attention ratio (sum) to source input")
plt.title(f'Attention distribution of ({model_shortname})', fontsize=16)
img_name = f'Prompt{args.prompt_id}_Prune{prune_ratio}.png'
plt.savefig(img_file_dir + img_name)
print(f"==>> saved to {img_file_dir + img_name}")

