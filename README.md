# IEEE-CIS Fraud Detection

Production-style fraud detection system dibangun di atas dataset IEEE-CIS (590K transaksi e-commerce). Proyek ini mencakup pipeline end-to-end mulai dari EDA, feature engineering, model training, imbalanced learning experiments, business-cost optimization, hingga REST API siap deploy dengan monitoring dashboard.

---

## Hasil Utama

| Metric | Value |
|---|---|
| AUC-ROC | **0.9609** |
| AUC-PR | **0.7951** |
| Recall @ threshold 0.093 | **80.1%** |
| Precision @ threshold 0.093 | **55.3%** |
| Business savings vs. no model | **$0.43M** (validation set) |

**Key finding:** Baseline LightGBM tanpa resampling mengungguli SMOTE, Undersampling, dan Class Weighting pada AUC-PR — metric yang tepat untuk dataset imbalanced (3.5% fraud rate).

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) — Vesta Corporation  
590,540 transaksi · ~3.5% fraud · 434 fitur mentah (transaction + identity)

```
data/raw/
  train_transaction.csv   # 652 MB — 590K rows × 394 cols
  train_identity.csv      #  25 MB — 144K rows × 41 cols
```

Download:
```bash
curl -L -o data/raw/train_transaction.csv \
  -H "Authorization: Bearer <YOUR_KAGGLE_TOKEN>" \
  "https://www.kaggle.com/api/v1/competitions/ieee-fraud-detection/data-download/train_transaction.csv"

curl -L -o data/raw/train_identity.csv \
  -H "Authorization: Bearer <YOUR_KAGGLE_TOKEN>" \
  "https://www.kaggle.com/api/v1/competitions/ieee-fraud-detection/data-download/train_identity.csv"
```

---

## Struktur Proyek

```
ieee-cis-fraud-detection/
├── data/
│   ├── raw/                  # CSV mentah dari Kaggle
│   └── processed/            # train_merged.parquet (post-merge)
│       └── features/         # train_features.parquet (post-FE)
├── models/
│   └── artifacts/
│       ├── lgbm_baseline.pkl       # trained model
│       ├── fe_artifacts.pkl        # freq_maps, agg_stats, medians
│       ├── feature_cols.json       # 419 feature names
│       ├── model_card.json         # metrics & threshold
│       └── experiment_comparison.json
├── notebooks/
│   ├── 01_problem_framing_dataset_audit.ipynb
│   ├── 02_eda_data_quality.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_imbalanced_learning.ipynb
│   └── 06_evaluation_risk_scoring.ipynb
├── src/
│   ├── data/
│   │   └── load_data.py            # load_raw(), merge_tables()
│   ├── features/
│   │   └── build_features.py       # build_all_features() — training & inference
│   └── api/
│       └── main.py                 # FastAPI app
├── scripts/
│   └── prepare_data.py             # merge CSV → parquet
├── reports/
│   └── dashboard.html              # monitoring dashboard (buka di browser)
├── tests/
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone & buat virtual environment
git clone <repo-url>
cd ieee-cis-fraud-detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. macOS Apple Silicon — LightGBM perlu libomp
brew install libomp
```

---

## Cara Menjalankan

### Pipeline penuh (urutan)

```bash
# Merge raw CSV → parquet
python scripts/prepare_data.py

# Jalankan notebook secara berurutan
jupyter lab
# buka notebooks/ dan run 01 → 06
```

### API

```bash
uvicorn src.api.main:app --reload
# → http://localhost:8000/docs  (Swagger UI)
```

**Contoh request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionDT": 86400,
    "TransactionAmt": 299.99,
    "ProductCD": "W",
    "card4": "visa",
    "P_emaildomain": "gmail.com"
  }'
```

**Response:**
```json
{
  "fraud_probability": 0.049910,
  "risk_bucket": "Low",
  "is_fraud": false,
  "threshold_used": 0.249,
  "latency_ms": 37.76
}
```

### Dashboard

```bash
cd reports && python -m http.server 3000
# buka http://localhost:3000/dashboard.html
```

---

## Roadmap & Notebook Summary

| # | Notebook | Yang Dilakukan |
|---|---|---|
| 01 | Problem Framing & Dataset Audit | Class imbalance analysis, missing value map, feature type breakdown |
| 02 | EDA & Data Quality | Distribusi fraud rate per fitur, korelasi, outlier detection |
| 03 | Feature Engineering | Time features, M-flag encoding, frequency encoding, aggregation, imputation |
| 04 | Baseline Model | LightGBM, 80/20 time-aware split, feature importance |
| 05 | Imbalanced Learning | Baseline vs. Class Weight vs. Undersampling vs. SMOTE |
| 06 | Evaluation & Risk Scoring | Threshold optimization, confusion matrix, risk buckets, business cost |

---

## Feature Engineering

419 fitur final dari 434 kolom mentah. Pipeline disimpan sebagai `fe_artifacts.pkl` dan dipakai ulang saat inference:

| Transform | Detail |
|---|---|
| Time features | `hour`, `day_of_week`, `is_night`, `is_weekend` dari `TransactionDT` |
| M-flag encoding | 9 kolom M1–M9 (T/F/nan) → integer |
| Frequency encoding | 6 high-cardinality categorical (`card1`, `card2`, `addr1`, dll.) |
| Aggregation features | mean/std TransactionAmt per card1, card2, addr1 |
| Identity flag | `has_identity` — apakah ada data identity |
| Median imputation | 374 kolom numerik dengan median dari training set |

---

## Imbalanced Learning Experiments

| Strategy | AUC-ROC | AUC-PR | F1 | Recall |
|---|---|---|---|---|
| **Baseline** ★ | **0.9609** | **0.7951** | 0.755 | 0.683 |
| Class Weighting | 0.9642 | 0.763 | 0.720 | 0.660 |
| Random Undersampling | 0.9618 | 0.778 | 0.736 | 0.681 |
| SMOTE | 0.9576 | 0.779 | 0.738 | 0.684 |

> **Insight:** LightGBM sudah cukup robust terhadap class imbalance. SMOTE justru menambah noise dari interpolasi di feature space yang tinggi dimensi, sehingga AUC-PR turun.

---

## Threshold Optimization

Default threshold 0.5 bukan pilihan terbaik untuk data imbalanced. Dua strategi diuji:

| Strategy | Threshold | Recall | Precision | F1 | Business Cost |
|---|---|---|---|---|---|
| Default (0.5) | 0.500 | 36.4% | 86.6% | 0.513 | $0.46M |
| F1-Maximization | 0.246 | 68.3% | 85.2% | 0.755 | $0.24M |
| **Cost Minimization** ★ | **0.093** | **80.1%** | **55.3%** | 0.654 | **$0.21M** |

Business cost model: FN cost = $500 (fraud lolos), FP cost = $10 (false alarm review).

---

## Risk Bucket System

| Bucket | Score Range | Action | Fraud Rate |
|---|---|---|---|
| Low | < 0.10 | Auto-approve | ~0.3% |
| Medium | 0.10 – 0.50 | Manual review | ~12% |
| High | ≥ 0.50 | Block | ~55% |

---

## API Endpoints

| Method | Endpoint | Deskripsi |
|---|---|---|
| GET | `/health` | Status model, threshold, n_features, metrics |
| POST | `/predict` | Score 1 transaksi |
| POST | `/predict/batch` | Score hingga 1000 transaksi |

Full interactive docs: `http://localhost:8000/docs`

---

## Tech Stack

- **Python 3.11** · pandas · numpy · scikit-learn
- **LightGBM** — gradient boosting dengan categorical support
- **imbalanced-learn** — SMOTE, RandomUnderSampler
- **FastAPI + uvicorn** — REST API async
- **joblib** — model & artifact serialization
- **pyarrow** — columnar storage (parquet)
- **Chart.js** — dashboard visualization

---

## Lessons Learned

1. **AUC-PR > AUC-ROC** untuk dataset imbalanced — ROC bisa menyesatkan karena TN besar
2. **Threshold bukan hyperparameter model** — harus dioptimasi terpisah berdasarkan business cost
3. **Feature engineering > model complexity** — 419 fitur engineered dari 434 raw memberi lebih banyak gain dibanding tuning LightGBM
4. **Inference pipeline harus identik dengan training** — simpan semua FE artifacts, jangan hitung ulang
5. **Resampling tidak selalu membantu** — untuk GBDT dengan dataset besar, class_weight seringkali cukup
