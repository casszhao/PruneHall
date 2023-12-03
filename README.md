# Pruning parameters is pruning hallucination


## Get pruned models
we prune models with Wanda and SparseGPT, which are both available in this [repo](https://github.com/locuslab/wanda). The virtual environment set up in the repo is also compatible with the rest experiments. 


## Evaluate hallucination

Installing/Using SummaC
```
pip install summac
```

Requirement issues: in v0.0.4, we've reduced package dependencies to facilitate installation. We recommend you install `torch` first and verify it works before installing `summac`.

The two trained models SummaC-ZS and SummaC-Conv are implemented in `model_summac` ([link](https://github.com/tingofurro/summac/blob/master/model_summac.py)). Once the package is installed, the models can be used like this:

### Example use

```python
from summac.model_summac import SummaCZS, SummaCConv

model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cpu") # If you have a GPU: switch to: device="cuda"
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu", start_file="default", agg="mean")

document = """Scientists are studying Mars to learn about the Red Planet and find landing sites for future missions.
One possible site, known as Arcadia Planitia, is covered instrange sinuous features.
The shapes could be signs that the area is actually made of glaciers, which are large masses of slow-moving ice.
Arcadia Planitia is in Mars' northern lowlands."""

summary1 = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers. This makes Arcadia Planitia ideal for future missions."
score_zs1 = model_zs.score([document], [summary1])
score_conv1 = model_conv.score([document], [summary1])
print("[Summary 1] SummaCZS Score: %.3f; SummacConv score: %.3f" % (score_zs1["scores"][0], score_conv1["scores"][0])) # [Summary 1] SummaCZS Score: 0.582; SummacConv score: 0.536

summary2 = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers."
score_zs2 = model_zs.score([document], [summary2])
score_conv2 = model_conv.score([document], [summary2])
print("[Summary 2] SummaCZS Score: %.3f; SummacConv score: %.3f" % (score_zs2["scores"][0], score_conv2["scores"][0])) # [Summary 2] SummaCZS Score: 0.877; SummacConv score: 0.709
```

We recommend using the SummaCConv models, as experiments from the paper show it provides better predictions. Two notebooks provide experimental details: [SummaC - Main Results.ipynb](https://github.com/tingofurro/summac/blob/master/SummaC%20-%20Main%20Results.ipynb) for the main results (Table 2) and [SummaC - Additional Experiments.ipynb](https://github.com/tingofurro/summac/blob/master/SummaC%20-%20Additional%20Experiments.ipynb) for additional experiments (Tables 1, 3, 4, 5, 6) from the paper.

### Evaluation Datasets
All data is in the data folder in json format. You do not need to preprocess it. 

SummaC Benchmark

The SummaC Benchmark consists of 6 summary consistency datasets that have been standardized to a binary classification task. The datasets included are:

<p align="center">
  <img width="500" src="https://tingofurro.github.io/images/tacl2021_summac_benchmark.png?1"><br />
  <b>% Positive</b> is the percentage of positive (consistent) summaries. IAA is the inter-annotator agreement (Fleiss Kappa). <b>Source</b> is the dataset used for the source documents (CNN/DM or XSum). <b># Summarizers</b> is the number of summarizers (extractive and abstractive) included in the dataset. <b># Sublabel</b> is the number of labels in the typology used to label summary errors.
</p>

The data-loaders for the benchmark are included in `benchmark.py` ([link](https://github.com/tingofurro/summac/blob/master/summac/benchmark.py)). Each dataset in the benchmark downloads automatically on first run. To load the benchmark:
```py
from summac.benchmark import SummaCBenchmark
benchmark_val = SummaCBenchmark(benchmark_folder="/path/to/summac_benchmark/", cut="val")
frank_dataset = benchmark_val.get_dataset("frank")
print(frank_dataset[300]) # {"document: "A Darwin woman has become a TV [...]", "claim": "natalia moon , 23 , has become a tv sensation [...]", "label": 0, "cut": "val", "model_name": "s2s", "error_type": "LinkE"}
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

If you'd like to contribute, or have questions or suggestions, you can contact us at phillab@berkeley.edu. All contributions welcome, for example helping make the benchmark more easily downloadable, or improving model performance on the benchmark.
