########### load dataset
import json
from typing import Any, Dict, List
from summac.benchmark import SummaCBenchmark
from loguru import logger

DATASETS = ["polytope", "factcc", "summeval"]

if __name__ == '__main__':

    logger.info('downloading datasets...')
    benchmark_val = SummaCBenchmark(benchmark_folder="data/temp_data", cut="val",  dataset_names=DATASETS)

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

        # clean
        dataset_uids = set([x[UID] for x in dataset])

        # non-dupped
        undupped = {}

        for indx, local_uid in enumerate(dataset_uids):

            point: Dict[str, Any] = [x for x in dataset if x[UID] == local_uid][0]

            undupped[f"{dataset_name}_{indx}"] = point

        with open(f'data/{dataset_name}.json', 'w', encoding='utf8') as f:
            json.dump(
                undupped,
                f,
                indent=4
            )


    logger.success("done!")