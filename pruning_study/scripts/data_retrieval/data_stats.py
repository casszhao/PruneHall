"""
collects and aggregates results accross prompts for a list of metrics
"""
import json
import os
from nltk import word_tokenize
import numpy as np

data_stats = {}

AVAIL_DATA = [
    'factcc',
    'legal_contracts',
    'polytope',
    'rct_summaries',
    'summeval'
]

if __name__ == '__main__':

    for dataset in AVAIL_DATA:

        # go down to the individual prompt results
        fname = os.path.join(
            'data',
            dataset + '.json'
        )

        COUNTER = 0

        with open(fname, 'r', encoding='utf8') as f:
            data = json.load(f)

        word_lengths = np.zeros(len(data))
        reference_lengths = np.zeros(len(data))

        for i, (key, value) in enumerate(data.items()):

            if isinstance(value['document'], str) is False:
                COUNTER += 1
                continue

            if len(value['document'].split()) < 5:
                COUNTER += 1
                continue

            reference = value.get("reference_summary", None) or value.get("claim", None) or value.get("target", None)

            word_lengths[i] = len(word_tokenize(value['document']))
            if reference is not None:
                reference_lengths[i] = len(word_tokenize(reference))


        data_stats[dataset] = {
            'dataset_size': len(data),
            'source_mean': round(word_lengths.mean(), 1),
            'source_max': round(word_lengths.max(), 1),
            "reference_mean": round(reference_lengths.mean(), 1),
            'reference_max': round(reference_lengths.max(), 1),
        }

    with open('data/stats.json', 'w', encoding='utf8') as f:
        json.dump(
            data_stats,
            f,
            indent=4,
            default=str
        )