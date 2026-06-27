# RetailRocket Recommender System — GRU4Rec

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![GPU](https://img.shields.io/badge/GPU-CUDA_12.8-orange.svg)

A session-based recommendation engine built on the [RetailRocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset), using a GRU4Rec deep learning model evaluated against a popularity baseline across 30+ production-grade metrics.

---

## Results at a Glance

> Evaluated on **333,952 test sessions** · **210,990 unique items** · **161,786 prefix queries**

### Ranking Metrics

| Metric | @5 | @10 | @20 |
|---|---|---|---|
| **Recall** | 0.3378 | 0.3951 | **0.4508** |
| **MRR** | 0.2481 | 0.2557 | **0.2596** |
| **NDCG** | 0.2704 | 0.2890 | **0.3030** |
| **Hit Rate** | 0.3378 | 0.3951 | **0.4508** |

### vs. Popularity Baseline (Recall@20)

| Model | Recall@20 | MRR@20 | NDCG@20 |
|---|---|---|---|
| **GRU4Rec** | **0.4508** | **0.2596** | **0.3030** |
| Popularity Baseline | 0.0106 | 0.0023 | 0.0041 |

GRU4Rec achieves **42.4× better recall** than the popularity baseline.

---

## Production Metrics

### Catalog & User Coverage

| Metric | Value |
|---|---|
| Catalog Coverage @20 | 28.08% |
| User-level Recall @20 | 49.12% |
| User-level MRR @20 | 30.49% |
| User-level NDCG @20 | 35.09% |
| Unknown Item Ratio | 0.0% |

### Item Popularity Segments (Cold-Start Analysis)

| Segment | Recall@20 | MRR@20 |
|---|---|---|
| Head (popular) | 0.6145 | 0.3657 |
| Medium | 0.2150 | 0.1018 |
| Tail (rare) | 0.0155 | 0.0045 |

The model performs strongly on head items and degrades gracefully on tail items — a typical pattern for session-based models. Cold-start on tail items is an area for future work.

### Session Length Breakdown (Recall@20)

| Session Length | Recall@20 |
|---|---|
| < 5 events | 0.4764 |
| 5–10 events | 0.3910 |
| 10–20 events | 0.3049 |
| 20+ events | 0.2313 |

Shorter sessions benefit most, as expected in next-item prediction.

### User Activity Breakdown (Recall@20)

| User Segment | Recall@20 |
|---|---|
| New users | 0.4649 |
| Medium activity | 0.3888 |
| Heavy users | 0.2702 |

### Diversity & Novelty

| Metric | Value |
|---|---|
| Intra-list Diversity | 0.6331 |
| Novelty (Info Content) | 19.1245 |
| Information Content | 12.7066 |
| Category Diversity | 6.7133 ✅ |
| Recommendation Stability (Jaccard) | 0.4843 |

**Category diversity is now 6.71** — recommendations are spread across different categories, showing strong cross-category exploration.

### Latency & System Resources

| Metric | Value |
|---|---|
| Avg Latency | 24.24 ms |
| P95 Latency | 44.02 ms |
| P99 Latency | 198.39 ms |
| Throughput | 81.2 prefixes/sec |
| Model Size | 236.94 MB |
| Peak GPU Memory | 1,270.05 MB |
| CPU RAM Usage | 4,633.87 MB |
| Eval Time (total) | ~33.2 minutes |

---

## Project Structure

```
RetailRocket Recommender System/
│
├── config.py               # Configuration & paths
├── preprocessing.py        # Data pipeline & feature engineering
├── train.py                # GRU4Rec training loop
├── main.py                 # Full evaluation entrypoint
├── evaluator.py            # Evaluation logic
├── dataloader.py           # Data loading utilities
├── metrics.py              # Core ranking metrics
├── business_metric.py      # Advanced production metrics
├── data_validation.py      # Data validation & integrity checks
├── pipeline.py             # End-to-end orchestration
│
├── app/
│   └── models.py           # GRU4Rec model definition
├── EDA.py                  # Exploratory data analysis
├── requirements.txt        # Python dependencies
├── .gitignore
│
├── data/
│   ├── raw/                # Raw CSVs (not tracked in git)
│   └── processed/          # Preprocessed data (not tracked)
│
├── models_official_final/  # Trained model checkpoints
├── outputs_official_final/ # Evaluation outputs
└── figures/                # EDA visualizations
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Dataset

The RetailRocket dataset is not included due to licensing. Download it from:  
**[Kaggle — RetailRocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)**

### 3. Place Raw Files

```
data/raw/
├── events.csv
├── item_properties_part1.csv
├── item_properties_part2.csv
└── category_tree.csv
```

---

## Running the Pipeline

### Option A — Full Pipeline (Recommended)

```bash
python pipeline.py
```

Runs preprocessing → training → evaluation in sequence.

---

### Option B — Step by Step

#### Step 1: Exploratory Data Analysis *(optional)*

```bash
python EDA.py
```

Generates visualizations to `figures/` and prints summary statistics.  
**Time:** ~2–5 min

---

#### Step 2: Preprocessing

```bash
python preprocessing.py
```

- Loads raw CSVs from `data/raw/`
- Applies a **temporal split** (80% train / 20% test) — no data leakage
- Engineers sessions, user features, and item features
- Builds item encoder from training data only

**Key outputs:**

```
data/processed/train/sessions.parquet
data/processed/test/sessions.parquet
data/processed/train/item_encoder.pkl
data/processed/train/gap_max.pkl
```

**Time:** ~5–10 min

---

#### Step 3: Training

```bash
python train.py
```

Trains GRU4Rec with:
- Early stopping (patience = 5)
- Learning rate scheduling
- Gradient clipping
- Label smoothing
- Temporal validation split (90% train / 10% val within training data)

**Key outputs:**

```
models_official_final/model_best.pt
models_official_final/config.json
models_official_final/training_history.parquet
```

**Time:** ~30–60 min (GPU recommended)

---

#### Step 4: Evaluation

```bash
python main.py
```

Evaluates GRU4Rec and the Popularity Baseline across 30+ metrics, prints a results table, and saves to `evaluation_results.json`.

**Time:** ~33.2 min · **Throughput:** ~81.2 prefixes/sec

---

## Why GRU4Rec?

GRU4Rec models user sessions as sequences, capturing temporal patterns and context (hour of day, day of week, time gaps between events) — making it ideal for next-item prediction in e-commerce where a user's current intent is best inferred from their recent behaviour, not their entire history.

Unlike matrix factorisation approaches, GRU4Rec works without persistent user IDs, handling anonymous and new users naturally. This matters on RetailRocket, where many sessions are one-off browsing events with no login.

---

## Model Architecture

**GRU4Rec** is a recurrent neural network designed for session-based recommendation. It encodes a user's interaction sequence (clicks, views, add-to-cart events) via a Gated Recurrent Unit and scores all candidate items to predict the next interaction.

| Hyperparameter | Value |
|---|---|
| Max sequence length | 50 |
| Batch size | 2,048 |
| Model size | 236.94 MB |

---

## Future Work

- **Tail item recall** is very low (1.5%). Auxiliary signals (item metadata, category hierarchy) or popularity-aware sampling could improve cold-start performance.
- **P99 latency** spikes to ~198 ms. Quantization or ONNX export could bring this down further for production serving.
- Cross-session user modelling to better serve heavy/returning users.



## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. The RetailRocket dataset is subject to its own terms on Kaggle.
