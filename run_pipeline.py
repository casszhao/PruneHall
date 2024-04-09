from typing import List
import torch
import numpy as np
from loguru import logger
import json, argparse

from tqdm import tqdm
from pruning_study.datamodels import Dataset, FinalResult, HallucinationResult, SummaryResult
from pruning_study.prompts import generate_prompt
from pruning_study.utils import (
    get_model_and_tokenzier,
    get_sequence_length,
    create_results_path,
    harmonize_data_format_and_add_prompts,
    batchify,
    save_results
)

from pruning_study.eval_funcs import ExperimentEvaluator

DEVICE = 'cpu'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path',
        type=str,
        help='The path that models are saved in, without the model itself'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        help='Model name'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Name of data file to use.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size. Setting to 1 as default as our gpus are not that powerful'
    )
    parser.add_argument(
        '--pruning-method',
        default="fullmodel",
        type=str,
        help='if using pruned model and which to use',
        choices=["fullmodel", "wanda", "sparsegpt", "magnitude"]
    )
    parser.add_argument(
        '--save-inbetween',
        default=True,
        type=bool,
        help='Whether to save each result the moment is produced or wait till the end',
    )
    parser.add_argument(
        '--prompt-id',
        default="A",
        type=str,
        choices=["A", "B", "C"],
        help='pick a prompt template from prompt list, A, B, C'
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    logger.info(
        f"Starting evaluation for {args.pruning_method} {args.model_name} on {args.dataset}"
    )

    # create results path
    result_path: str = create_results_path(
        model_name=args.model_name,
        pruning_method=args.pruning_method,
        dataset=args.dataset
    )

    logger.info(f"Results to be saved at {result_path}")

    # get model and tokenizer
    model, tokenizer = get_model_and_tokenzier(
        model_name=args.model_name,
        cache_dir=args.model_path,
        torch_dtype=None,
        device_map=DEVICE
    )

    # set model to eval
    model.to(DEVICE)
    model.eval()

    logger.debug(f"Model and tokenizer succesfully loaded from {args.model_path}")

    # get maximum allowable seq lenght
    max_sequence_length: int = get_sequence_length(model.config)

    # get evaluation functions
    evaluation_functions = ExperimentEvaluator(
        device=DEVICE
    )

    logger.debug("Evaluation scripts loaded - starting..")

    with open(f"data/{args.dataset}.json", 'r', encoding='utf8') as f:
        data_untreated = json.load(f)

    dataset: Dataset = harmonize_data_format_and_add_prompts(
        data_untreated=data_untreated,
        model_type = args.model_name.split("-")[0].lower(),
        prompt_id=args.prompt_id
    )

    # collect the results
    full_results = {}

    # evaluation
    with torch.no_grad():
        for indx, datapoints in tqdm(enumerate(batchify(dataset, batch_size=args.batch_size))):

            # get only the prompts
            prompts = [x.prompt for x in datapoints]

            # padding side is always left for batch generation decoder-only
            tokenizer.padding_side = "left"

            tokenizer.pad_token = tokenizer.eos_token # JUST FOR QUICK LOCAL TESTS WITH GPT2

            # get model inputs
            model_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True
            )
            model_inputs = model_inputs.to(DEVICE)

            # just an assertion on lengths to not surpass model length
            if len(model_inputs['input_ids']) > max_sequence_length:
                raise ValueError("Should never happen; we tested it on all datasets")

            # forward loops
            output = model.generate(
                **model_inputs,
                max_length=max_sequence_length,
                do_sample = False
            )

            # needed to remove the prompts
            # **IMPORTANT STEP**
            # since we padded all inputs are the same length
            output_to_decode = output[:,model_inputs['input_ids'].size(1):].cpu()
            # decode the outputs
            predictions = tokenizer.batch_decode(
                output_to_decode,
                skip_special_tokens=True
            )

            # evaluate summary / get reference
            target_summaries: List[str] = [x.target_summary for x in datapoints]

            summary_evaluations: SummaryResult = evaluation_functions.evaluate_summary(
                prediction=predictions, reference=target_summaries
            )

            # evaluate hallucinations
            documents: List[str] = [x.document for x in datapoints]

            hallucination_results: HallucinationResult = evaluation_functions.evaluate_hallucunations(
                prediction=predictions,
                reference=documents
            )

            # empty any garbage
            torch.cuda.empty_cache()

            # collect the results
            for point_indx, datapoint in enumerate(datapoints):

                full_results[datapoint.id] = FinalResult(
                    id= datapoint.id,
                    document= datapoint.document,
                    generated= predictions[point_indx],
                    rouge={
                        'rouge1': summary_evaluations.rouge['rouge1'][point_indx],
                        'rouge2': summary_evaluations.rouge['rouge2'][point_indx],
                        'rougeL': summary_evaluations.rouge['rougeL'][point_indx],
                    },
                    bertscore={
                        'precision': summary_evaluations.bertscore['precision'][point_indx],
                        'recall': summary_evaluations.bertscore['recall'][point_indx],
                        'f1': summary_evaluations.bertscore['f1'][point_indx],
                    },
                    summac_zs=hallucination_results.summac_zs[point_indx],
                    harim_plus=hallucination_results.harim_plus[point_indx],
                    summac_conv=hallucination_results.summac_conv[point_indx]
                ).model_dump()

                # save them intermedietary just for inspection and debugging
                # + in case of any cuda mem failure
                if args.save_in_the_middle:
                    save_results(
                        results=full_results,
                        results_path=result_path,
                        prompt_id=args.prompt_id
                    )

            if indx == 1:
                break
    # save final results
    save_results(
        results=full_results,
        results_path=result_path,
        prompt_id=args.prompt_id
    )

    logger.success("Finished!")