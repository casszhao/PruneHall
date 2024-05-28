import json
import os
from typing import Any, Dict, Generator, List, Optional, Tuple
from loguru import logger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from pruning_study.datamodels import Datapoint, Dataset
from pruning_study.prompts import generate_prompt

def get_sequence_length(config: PretrainedConfig, default: int = 2048) -> int:
    """
    gets maximum allowable sequence length based on positional embeddings
    """
    if hasattr(config, "sliding_window"):
        return config.sliding_window
    elif hasattr(config, "max_position_embeddings"):
        return config.max_position_embeddings
    else:
        return default


def get_model_and_tokenzier(
        model_name: str,
        cache_dir: str = "llm_weights",
        torch_dtype: Optional[str] = None,
        device_map: Optional[str] = None
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """loads the model and tokenizer"""

    if torch_dtype is None:
        torch_dtype = "auto" if torch.cuda.is_available() else None
    if device_map is None:
        device_map = 'auto'

    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(cache_dir, model_name),
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map=device_map
    )

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(cache_dir, model_name),
        use_fast=False,
        cache_dir=cache_dir
    )

    return model, tokenizer


def create_results_path(
        model_name: str,
        pruning_method: str,
        dataset: str,
        results_path: str = 'generated_output',
    ) -> str:
    """creates the result path"""

    save_dir: str = os.path.join(
        results_path,
        model_name,
        pruning_method,
        dataset,
        ""
    )

    os.makedirs(save_dir, exist_ok=True)

    return save_dir

def save_results(
    results: Dict[str, Any],
    results_path: str,
    prompt_id: str
) -> None:
    """saves the results"""

    try:

        fname = os.path.join(results_path, 'prompt_' + prompt_id + '.json')
        with open(fname, 'w', encoding='utf8') as f:
            json.dump(
                obj=results,
                fp=f,
                indent=4,
                default=str
            )

        return
    except Exception as exc:
        message = f"Failed when saving results \n {repr(exc)}"
        logger.error(message)
        raise exc

def harmonize_data_format_and_add_prompts(
        data_untreated: Dict[str, Any],
        model_type: str,
        prompt_id: str
    ) -> Dataset:
    """gets untreated data and returns a structured dataset"""

    treated_data: Dataset = []

    # loop through each untreated dataset
    for uid, values in data_untreated.items():

        try:

            # get the target summary
            target_summary: str | None = (
                values.get("reference_summary")
                or values.get("claim")
                or values.get("target")
            )

            # if its none no point
            if target_summary is None:
                raise ValueError

            # get the prompt for it
            prompt: str = generate_prompt(
                task="summarization",
                model=model_type,
                prompt_id=prompt_id,
                document=values['document'],
            )

            # create the harmonizaed datapoint
            treated_data.append(
                Datapoint(
                    id=uid,
                    document=values['document'],
                    target_summary=target_summary,
                    prompt=prompt
                )
            )
        except ValueError:
            message = f"{uid} has no target summary under \
                `reference_summary` , `claim` or `target`. \
                    skipping.. \n"
            logger.error(message)
            continue

    return treated_data


def batchify(iterable: List[Any], batch_size: int = 32) -> Generator[Any, None, None]:
    """batches over an iterable"""
    if iterable == []:
        return []
    if batch_size < 1:
        raise ValueError("`batch_size` must be at least 1")
    iterable_len = len(iterable)
    for batch_start_indx in range(0, iterable_len, batch_size):
        yield iterable[
            batch_start_indx : min(batch_start_indx + batch_size, iterable_len)
        ]
