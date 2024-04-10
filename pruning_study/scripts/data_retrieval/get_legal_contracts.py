import json
from typing import Any, Dict


if __name__ == '__main__':

    # taken from `https://github.com/lauramanor/legal_summarization/blob/master/tldrlegal_v1.json`
    with open('data/temp_data/legal/tldrlegal_v1.json', 'r', encoding='utf8') as f:
        data = json.load(f)

    legal_data: Dict[str, Dict[str, Any]] = {}

    for uid, values in data.items():

        legal_data[uid] = {
            'uid': values['uid'],
            'id': values['id'],
            'dataset': 'tldrlegal_v1',
            'document': values['original_text'],
            'target': values['reference_summary'],
            'title': values['title']
        }

    with open('data/legal_contracts.json', 'w', encoding='utf8') as f:
        json.dump(
            legal_data,
            f,
            indent=4,
        )