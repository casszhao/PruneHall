
# Investigating Hallucinations in Pruned Large Language Models for Abstractive Summarization
Implementation of evaluating hallucination of pruned models, as presented in our [paper](https://arxiv.org/pdf/2311.09335.pdf).

Despite the remarkable performance of generative large language models (LLMs) on abstractive summarization, they face two significant challenges: their considerable size and tendency to hallucinate. Hallucinations are concerning because they erode reliability and raise safety issues. Pruning is a technique that reduces model size by removing redundant weights, enabling more efficient sparse inference. Pruned models yield downstream task performance comparable to the original, making them ideal alternatives when operating on a limited budget.

However, the effect that pruning has upon hallucinations in abstractive summarization with LLMs has yet to be explored. In this paper, we provide an extensive empirical study across five summarization datasets, two state-of-the-art pruning methods, and five instruction-tuned LLMs.

Surprisingly, we find that hallucinations from pruned LLMs are less prevalent than the original models. Our analysis suggests that pruned models tend to depend more on the source document for summary generation. This leads to a higher lexical overlap between the generated summary and the source document, which could be a reason for the reduction in hallucination risk.

## Model pruning
We prune models using Magnitude, Wanda and SparseGPT, using the default code available as is found in the [Wanda repo](https://github.com/locuslab/wanda).
Pruning is done separately in the suggested environment by the previous repo and as such the suggested dependencies with our repo do not necessarily agree.

## Setting up

You need to have poetry installed. If not you should be able to install poetry using:

```bash
pipx install poetry
```

else, please follow [poetry's original documentation](https://python-poetry.org/docs/).

Following this you can follow these steps to set-up your environment:

```bash
poetry shell
poetry install
# install summac separately due to dependency clashes
pip install summac
```

## Evaluation

Our `run_pipeline.py` script does three main things:
1. Generates in `eval` and without `sampling` a summary for a document based on a prompt.
2. Evaluates the summary based on `Rouge metrics` and `BertScore` against the target summary.
3. Evaluates for hallucinations the summary based on `SummaC` and `Harim+` against the source document.

For our work we also produce `rouge metrics` of the summary against the source document for our across sparsity experiments.

You can run our scripts by executing the following script:

```bash
python run_pipeline.py  --model-path _PATH_WHERE_YOUR_MODELS_ARE_STORED \
                        --model-name _MODEL_NAME_AS_IT_APPEARS_IN_YOUR_DIR_ \
                        --data-path _PATH_WHERE_YOUR_DATA_IS_STORED_ \
                        --dataset _DATASET_NAME_ \
                        --seed _RANDOM_SEED_FOR_EXPERIMENTS_ \
                        --batch-size _EVAL_BATCH_SIZE \
                        --pruning-method _PRUNING_METHOD_SO_WE_CAN_SAVE_RESULTS_ \
                        --save-inbetween _WHETHER_TO_STORE_INTERMEDIATE_RESULTS_ \
                        --prompt-id _WHICH_PROMPT_TEMPLATE_TO_USE_
```

Example:

```bash
python run_pipeline.py  --model-path models \
                        --model-name llama-2-7b-chat-wanda # note under here your tokenizer and model are both saved \
                        --data-path data/ \
                        --dataset 'summeval' \
                        --seed 0 \
                        --batch-size 16 \
                        --pruning-method 'wanda' \
                        --save-inbetween true \
                        --prompt-id 'A'
```

## Cite us
```
@article{chrysostomou2023lighter,
  title={Lighter, yet More Faithful: Investigating Hallucinations in Pruned Large Language Models for Abstractive Summarization},
  author={Chrysostomou, George and Zhao, Zhixue and Williams, Miles and Aletras, Nikolaos},
  journal={arXiv preprint arXiv:2311.09335},
  year={2023}
}
```

## Contributing

If you'd like to contribute or have questions or suggestions, you can contact us at zhixue.zhao@sheffield.ac.uk

All discussion and contributions are welcome.