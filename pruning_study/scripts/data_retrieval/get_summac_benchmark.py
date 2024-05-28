########### load dataset
import json
from typing import Any, Dict, List
from summac.benchmark import SummaCBenchmark
from loguru import logger
from datasets import load_dataset

DATASETS = ["polytope", "factcc", "summeval"]

def get_reference_summary(summac_datapoint: dict, CNNDM_test: dict) -> str:
    """gets the target sumamry from the data"""

    _, _, id = summac_datapoint['cnndm_id'].split('-')

    return CNNDM_test[id]['highlights']

if __name__ == '__main__':

    logger.info('downloading datasets...')
    benchmark_val = SummaCBenchmark(benchmark_folder="data/temp_data", cut="val",  dataset_names=DATASETS)

    CNNDM = load_dataset("cnn_dailymail", "3.0.0")
    # convert to dict for easier search
    CNNDM_test = {v['id']: v for v in CNNDM['test']}

    for _, dataset_name in enumerate(DATASETS):

        if dataset_name == 'factcc':
            UID = 'id'
        elif dataset_name == 'polytope':
            UID = 'ID'
        elif dataset_name == 'summeval':
            UID = 'cnndm_id'
        else: raise NotImplementedError

        logger.info(f'Formatting {dataset_name}..')

        dataset: List[Dict[str, Any]] = benchmark_val.get_dataset(dataset_name=dataset_name)
        # just to clean it up a bit
        skimmed_down_dataset: List[Dict[str, Any]] = []
        # for summeval get human reference summary
        if dataset_name == 'summeval':
            for i, datapoint in enumerate(dataset):
                skimmed_down_dataset.append({
                    UID: datapoint[UID],
                    'document' : datapoint['document'],
                    'target': get_reference_summary(datapoint, CNNDM_test=CNNDM_test)
                })
        else:
            for i, datapoint in enumerate(dataset):
                skimmed_down_dataset.append({
                    UID: datapoint[UID],
                    'document' : datapoint['document'],
                    'claim': datapoint['claim']
                })

        # clean
        dataset_uids = set([x[UID] for x in skimmed_down_dataset])

        # non-dupped
        undupped = {}

        for indx, local_uid in enumerate(dataset_uids):

            point: Dict[str, Any] = [x for x in skimmed_down_dataset if x[UID] == local_uid][0]

            undupped[f"{dataset_name}_{indx}"] = point

        with open(f'data/{dataset_name}.json', 'w', encoding='utf8') as f:
            json.dump(
                undupped,
                f,
                indent=4
            )


    logger.success("done!")