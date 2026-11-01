# Datasets and expected inputs

## Sources

- Amazon CDs and Vinyl, Toys and Games, and Video Games: [Amazon Review Data (2018)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).
- Yelp: [Yelp Open Dataset](https://www.yelp.com/dataset).
- The archived Toys metadata also includes parquet files from McAuley-Lab/Amazon-Reviews-2023. See the [official dataset documentation](https://amazon-reviews-2023.github.io/main.html). Do not assume all interaction and metadata files use the same release.

The experiments use processed subsets, not the complete source datasets. CDs and Yelp interaction inputs were retained upstream `.inter` tables; reconstructing them from current raw downloads is not guaranteed to yield identical subsets. Toys and Video Games sources already contain full-period 5-core filtering.

## Input contract

- Item IDs: 1..N; ID 0 is padding. The same mapping must index sequences, targets and features.
- Histories: integer tensor [B,L], L<=50, nonempty, valid IDs first and zeros on the right.
- Targets: IDs in 1..N for full-catalog cross-entropy; logit column = item ID minus one.
- Text features: float32 [N+1,384], frozen, attention-mask mean pooling followed by L2 normalization with all-MiniLM-L6-v2; maximum text tokens 128.
- Metadata availability: boolean [N+1], false for missing features and padding.
- Amazon text fields: title, brand, category, description. Yelp fields: name, categories, city, state. Archived text construction truncates concatenated text to 1,600 characters. Some historical features are inherited caches; re-encoding can change inputs.

## Frozen experiment protocol

| Dataset | Training users | Training items | Training prefixes | Test users |
|---|---:|---:|---:|---:|
| CDs | 4000 | 15576 | 93306 | 662 |
| Yelp | 4000 | 31157 | 80771 | 1527 |
| Toys | 4000 | 19305 | 24547 | 703 |
| Video Games | 35530 | 13577 | 232174 | 4939 |

CDs/Yelp/Toys use training before q60, validation in q60–q70, and test targets at or after q90. Video Games uses training before q80, validation in q80–q90, and test at or after q90. These are the quantiles of each archived corpus, not universal dates. Histories precede their targets. Upstream filtering and metadata snapshot timing remain limitations.

Evaluation ranks the training item vocabulary, filters seen items, and retains out-of-vocabulary test targets in the all-user denominator with zero credit. Report Hit/NDCG at 5/10/20/50/100; select settings and checkpoints by validation NDCG@100. Trainable methods use seeds 42, 2026 and 3407. Full split/target selection, feature preparation and evaluation scripts are outside this core release; the rules above alone do not reconstruct the exact frozen experiment.
