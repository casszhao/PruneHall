import argparse
from huggingface_hub import snapshot_download

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local-dir',
        type=str,
        help='The path to save the model'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        help='HF repo id'
    )

    args = parser.parse_args()

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir
    )