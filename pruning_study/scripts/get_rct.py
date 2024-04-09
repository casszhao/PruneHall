import json
from typing import Any, Dict, List
import pandas as pd

DATA_PATH = "data/"

if __name__ == '__main__':

    # data downloaded from `https://github.com/bwallace/RCT-summarization-data/blob/main/`
    test_inputs = pd.read_csv('data/temp_data/rct/test-inputs.csv')
    test_targets = pd.read_csv('data/temp_data/rct/test-targets.csv')

    final_data: Dict[str, Dict[str, Any]] = {}

    unique_reviews = set()

    for row in test_targets.to_dict('records'):

        matched_articles: List[Dict[str, Any]] = \
            test_inputs[test_inputs.ReviewID == row['ReviewID']].to_dict('records')

        # only interested in 1 to 1 matching
        if len(matched_articles) > 1:
            continue
        if len(matched_articles) == 0:
            continue

        matched_article = matched_articles[0]

        unique_reviews.add(row['ReviewID'])
        final_data[row['ReviewID']] = {
            'review_id': row['ReviewID'],
            'pmid': matched_article['PMID'],
            'title': matched_article['Title'],
            'document': matched_article['Abstract'],
            'dataset': 'rct_summaries',
            'target': row['Target']
        }


    assert len(final_data) == len(unique_reviews)

    with open('data/rct_summaries.json', 'w', encoding='utf8') as f:

        json.dump(
           final_data,
           f,
           indent = 4
        )

