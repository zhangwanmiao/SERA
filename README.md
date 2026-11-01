# Semantic-Enhanced Sequential Recommendation with Recent-State Aggregation

Core implementation for a study of frozen-text semantic enhancement and recent-state Mean/LSE scoring. The paper is in preparation; SERA is a provisional short name.

## Model

Frozen 384-dimensional MiniLM item features are projected into 128 dimensions and added to item ID embeddings. A two-layer causal Transformer encodes histories. Recent valid states score each candidate, and Mean plus normalized temperature-scaled LSE produces the final score. History and candidate embeddings use the same enhanced item table.

- `sera/semantic.py`: semantic residual enhancement.
- `sera/encoder.py`: causal history encoding.
- `sera/matching.py`: recent-state selection and candidate matching.
- `sera/pooling.py`: valid-state Mean/LSE aggregation.
- `sera/model.py`: compact model assembly.
- `configs/`: frozen main-model settings for four datasets.

## Setup and example

Install a PyTorch build appropriate for your device, then:

```bash
python -m pip install -r requirements.txt
python example.py
```

The example uses synthetic inputs and performs one training step. It is not a benchmark reproduction command. The synthetic example was verified on CPU in an isolated Python 3.11 environment with PyTorch 2.15.0, NumPy 1.26.4 and RecBole 1.2.1. It produced logits of shape (2, 20) and completed one forward/backward optimizer step. This check does not validate benchmark metrics.

## Data

Experiments use Amazon CDs and Vinyl, Toys and Games, Video Games, and Yelp. See [DATASETS.md](DATASETS.md) for sources, input contracts and evaluation boundaries. Download data from the original providers under their terms; data and pretrained encoder weights are not included.

## Release scope

The four component functions are copied from the archived research modules. The compact wrapper is newly assembled from the same computation. This release does not include the original preprocessing, tuning, baseline training, checkpoints or locked evaluation pipeline and has not been validated to reproduce the reported metrics. Full experimental artifacts or further preparation guidance can be discussed via a GitHub issue after publication of this repository.

## Acknowledgments

The causal backbone uses the SASRec-style implementation provided by [RecBole](https://github.com/RUCAIBox/RecBole). Text features use [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). These dependencies retain their respective licenses. A license for this repository has not yet been selected.
