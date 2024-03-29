import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from datasets import load_dataset,load_from_disk, list_metrics # metrics_list = list_metrics()
from evaluate import load
from summac.model_summac import SummaCZS, SummaCConv
from summac.benchmark import SummaCBenchmark
import nltk
import pandas as pd
import numpy as np
import os, json, argparse, time
from prompt_functions import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Model name or path.')
parser.add_argument('--data', type=str, help='Name of data file to use.')
parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
parser.add_argument('--prune_method', default="fullmodel", type=str, help='if using pruned model and which to use', 
                    choices=["fullmodel", "wanda", "sparsegpt", "magnitude"])
parser.add_argument('--prompt_id', default="A", type=str, 
                    choices=["A", "B", "C"],
                    help='pick a prompt template from prompt list, A or B or None')
args = parser.parse_args()



np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

short_model_name = str(args.model).split("/")[-1]
short_model_type = short_model_name.split("-")[0].lower()
save_path = os.path.join("generated_output", short_model_name, args.prune_method, args.data)
os.makedirs(save_path, exist_ok=True)


def get_sequence_length(config: PretrainedConfig, default: int = 2048) -> int:
    if hasattr(config, "sliding_window"):
        return config.sliding_window
    elif hasattr(config, "max_position_embeddings"):
        return config.max_position_embeddings
    else:
        return default


def get_model_tokenzier(model_name, cache_dir = "llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype="auto" if torch.cuda.is_available() else None,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=False,
        cache_dir=cache_dir
    )

    return model, tokenizer


# ########### load metrics
harim = load("NCSOFT/harim_plus")  #  using model : facebook/bart-large-cnn
rouge = load("rouge")
bertscore = load("bertscore")
model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cuda", use_con=False) # If you have a GPU: switch to: device="cuda"-
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")
########### load model
model_path = args.model
if args.prune_method != "fullmodel":
    short_name = str(args.model).split("/")[-1]
    model_path = f"pruned_model/{short_name}/{args.prune_method}"

model, tokenizer = get_model_tokenzier(model_path)
model.eval()

sequence_length = get_sequence_length(model.config)

########### load dataset
# benchmark_val = SummaCBenchmark(benchmark_folder="data/", cut="val") 
# dataset = benchmark_val.get_dataset(args.data) 
f = open(f"data/{args.data}.json")
dataset = json.load(f)
# summeval  dict_keys(['document', 'claim', 'label', 'model_name', 'cnndm_id', 'cut', 'annotations', 'dataset', 'origin', 'error_type'])
# factcc    dict_keys(['claim', 'label', 'filepath', 'id', 'document', 'annotations', 'dataset', 'origin'])
# polytope  dict_keys(['ID', 'document', 'claim', 'errors', 'cut', 'overall_label', 'omission_label', 'addition_label', 'duplication_label', 'inaccuracy_label', 'label', 'dataset', 'annotations', 'origin'])
# xsumfaith dict_keys(['document', 'claim', 'bbcid', 'model_name', 'label', 'cut', 'annotations', 'dataset', 'origin'])

key_list = list(dataset.keys())

for i, key in enumerate(key_list):
    # prepare full input based on prompt template
    if i == 0: 
        try: 
            with open(save_path + f"/prompt_{args.prompt_id}_raw_result.json", "r+") as json_file:
                generate_dict = json.load(json_file)
                print(' \n \n \n')
                print("".center(50, "-"))
                print(' countinue from last time')
        except: generate_dict = dataset.copy()

    if 'generated' in list(generate_dict[key].keys()):
        print(key)
    else: 
        print(f"generating for {key} ...")
        document = dataset[key]["document"]
        prompt = generate_prompt(
            task="summarization",
            model=short_model_type,
            prompt_id=args.prompt_id,
            document=dataset[key]["document"],
        )
        #character_len = len(dataset[key]['document'])

        encoding = tokenizer(prompt, return_tensors="pt")
        encoding = encoding.to(model.device)

        input_ids = encoding.input_ids
        if input_ids.shape[1] > sequence_length:
            print(f"Skipping {key} ({input_ids.shape[1]} > {sequence_length}).")
            continue

        output = model.generate(
            **encoding,
            max_length=sequence_length,
            do_sample=False,
        )
        output_text = tokenizer.decode(output[0][int(input_ids.shape[1]):], skip_special_tokens=True)
        
        reference = (
            dataset[key].get("reference_summary")
            or dataset[key].get("claim")
            or dataset[key].get("target")
        )
        if reference is None:
            raise ValueError(f"Could not find a reference for {key}.")

        score_harim = harim.compute(predictions=[output_text], references=[document])
        score_rouge = rouge.compute(predictions=[output_text], references=[reference])
        score_bertscore = bertscore.compute(predictions=[output_text], references=[reference], lang="en")
        score_zs = model_zs.score([document], [output_text])
        score_conv = model_conv.score([document], [output_text])


        generate_dict[key]['generated'] = output_text
        generate_dict[key]['rouge'] = score_rouge
        generate_dict[key]['bertscore'] = score_bertscore
        generate_dict[key]['harim'] = score_harim
        generate_dict[key]['summac_conv'] = score_conv["scores"][0]
        generate_dict[key]['summac_zs'] = score_zs["scores"][0]


        ######### this part is only for quick testing and saving
        json_object = json.dumps(generate_dict, indent=4)
        with open(save_path + f"/prompt_{args.prompt_id}_raw_result.json", "w") as outfile:
            outfile.write(json_object)
            outfile.close()
            print(key)
        ######### this part is only for quick testing and saving
        #torch.cuda.empty_cache()


json_object = json.dumps(generate_dict, indent=4)
with open(save_path + f"/prompt_{args.prompt_id}_raw_result.json", "w") as outfile:
    outfile.write(json_object)
    outfile.close()