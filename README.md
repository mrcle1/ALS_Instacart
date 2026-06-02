# Instacart Recommender

End-to-end recommendation pipeline for the Instacart Superstore dataset.

Refactored from the original Databricks notebook into a modular Python package with:

- ALS collaborative filtering
- FP-Growth association-rule recommendations
- Parallel evaluation using Joblib
- Training and inference visualizations
- FastAPI serving layer

---

## Project Structure

```text
instacart_recommender/
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│       ├── health.py
│       ├── train.py
│       ├── recommend.py
│       └── metrics.py
│
├── src/
│   ├── config.py
│   ├── logger.py
│   ├── progress.py
│   │
│   ├── data/
│   │   ├── loader.py
│   │   ├── splitter.py
│   │   └── encoders.py
│   │
│   ├── models/
│   │   ├── als_model.py
│   │   └── fpgrowth_model.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluator.py
│   │
│   ├── visualization/
│   │   ├── _style.py
│   │   ├── training_viz.py
│   │   └── inference_viz.py
│   │
│   └── pipeline/
│       ├── train_pipeline.py
│       └── infer_pipeline.py
│
├── artifacts/
│   ├── models/
│   ├── plots/
│   ├── logs/
│   └── predictions/
│
├── config.ini
├── pipeline.yml
├── requirements.txt
├── run.py
└── README.md
```

### Directory Overview

| Path | Purpose |
|--------|---------|
| `api/` | FastAPI application |
| `src/data/` | Data loading, splitting, and encoding |
| `src/models/` | Recommendation algorithms |
| `src/evaluation/` | Ranking metrics and evaluation |
| `src/visualization/` | Training and inference dashboards |
| `src/pipeline/` | Training and inference orchestration |
| `artifacts/` | Generated models, plots, logs, and predictions |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the Dataset

Place the Instacart dataset at the location specified in `config.ini`.

Example:

```bash
ls /data/instacart/Instacart_TrainData.parquet
```

### 3. Train Models

```bash
python run.py train
```

### 4. Run Inference

```bash
python run.py infer --algo als --N 10

python run.py infer --algo fpgrowth --N 10
```

### 5. Launch the API

```bash
python run.py serve
```

Open:

```text
http://localhost:8000/docs
```

---

## Pipeline Stages

The `pipeline.yml` file defines the complete training and inference workflow.

| Stage | Module | Purpose |
|---------|---------|---------|
| `ingest_raw` | `src.data.loader` | Load source dataset |
| `split` | `src.data.splitter` | Per-user train/test split |
| `encode` | `src.data.encoders` | Encode users and products |
| `build_matrices` | `src.data.encoders` | Create sparse user-item matrix |
| `build_ground_truth` | `src.data.encoders` | Generate evaluation targets |
| `train_als` | `src.models.als_model` | Train ALS recommender |
| `train_fpgrowth` | `src.models.fpgrowth_model` | Train FP-Growth recommender |
| `evaluate_als` | `src.evaluation.evaluator` | Compute ranking metrics |
| `render_training_viz` | `src.visualization.training_viz` | Generate training dashboard |
| `infer_als` | `src.pipeline.infer_pipeline` | ALS recommendations |
| `infer_fpgrowth` | `src.pipeline.infer_pipeline` | FP-Growth recommendations |

---

## Progress Bars

Project-wide progress bars are provided by:

```python
from src.progress import instacart_tqdm
```

Example:

```python
for item in instacart_tqdm(items, desc="training"):
    ...
```

Default configuration:

```text
Color : #05ad46
Width : 200 columns

{desc:30}: {percentage:3.0f}%|{bar:60}| {n_fmt}/{total_fmt}
[{elapsed}<{remaining}, {rate_fmt}]
```

---

## Parallel Processing

The project uses `joblib.Parallel(n_jobs=-1)` for CPU-intensive workloads.

Implemented in:

- `src.data.splitter.build_train_test_split_parallel`
- `src.evaluation.evaluator.evaluate_recommender`
- `src.models.als_model.ALSRecommender.recommend_batch`
- `src.models.fpgrowth_model.FPGrowthRecommender.recommend_batch`

---

## Logging

Use the project logger instead of `print()`:

```python
from src.logger import get_logger

log = get_logger(__name__)
```

Supported levels:

```python
log.info(...)
log.warning(...)
log.error(...)
log.exception(...)
```

Log output is written to:

```text
artifacts/logs/instacart_recommender.log
```

and streamed to standard error.

---

## API Endpoints

| Method | Endpoint | Description |
|----------|------------|-------------|
| `GET` | `/health` | Service status and artifact availability |
| `POST` | `/train` | Execute training pipeline |
| `POST` | `/recommend` | Generate recommendations |
| `GET` | `/metrics` | Retrieve evaluation metrics |

### Request Examples

#### Train

```json
{
  "raw_path": "...",
  "config_path": "..."
}
```

#### Recommend

```json
{
  "algorithm": "als",
  "user_ids": [1, 2, 3],
  "N": 10
}
```

### OpenAPI Documentation

Available when the API is running:

```text
http://localhost:8000/docs
```