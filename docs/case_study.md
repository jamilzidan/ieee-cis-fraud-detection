# Case Study: IEEE-CIS Fraud Detection

**End-to-end fraud detection system — IEEE-CIS Dataset**

---

## Latar Belakang

Vesta Corporation, perusahaan payment services, menyediakan dataset transaksi e-commerce dari sistem fraud prevention mereka. Dataset ini dirilis di Kaggle sebagai IEEE-CIS Fraud Detection Competition.

**Konteks bisnis:** Setiap transaksi yang lolos tapi sebenarnya fraud merugikan perusahaan sekitar $500 (kerugian langsung + chargeback). Sebaliknya, setiap transaksi legit yang diblok merugikan sekitar $10 (opportunity cost + customer friction). Artinya, membiarkan fraud lolos 50x lebih mahal daripada false alarm.

**Tujuan proyek ini:** Bangun sistem yang bisa membedakan transaksi fraud dari yang legit, lalu deploy sebagai API siap pakai — bukan sekadar notebook Kaggle.

---

## Step 1 — Problem Framing & Dataset Audit

### Kenapa step ini penting
Sebelum nulis satu baris kode modeling, perlu dipahami dulu karakteristik data. Keputusan di sini menentukan semua langkah berikutnya.

### Yang ditemukan
Dataset terdiri dari dua tabel:
- `train_transaction.csv` — 590.540 baris, 394 kolom, 652 MB
- `train_identity.csv` — 144.233 baris, 41 kolom (hanya ~24% transaksi punya data identity)

Setelah merge LEFT JOIN on `TransactionID`, hasilnya 590.540 baris × 434 kolom.

**Temuan kritis — class imbalance:**
```
Non-fraud : 569.877 (96.5%)
Fraud     :  20.663  (3.5%)
```
Dari 590K transaksi, hanya 3.5% yang fraud. Ini bukan sekadar "sedikit imbalance" — kalau model bodoh yang selalu predict "bukan fraud" pun bisa dapat akurasi 96.5%. Ini berarti **akurasi bukan metric yang tepat**.

**Temuan lain:**
- Kolom V1–V339 adalah fitur proprietary Vesta, artinya interpretasinya terbatas
- Banyak kolom dengan missing value >90% (perlu di-drop saat feature engineering)
- `TransactionDT` bukan Unix timestamp biasa — ini delta detik dari titik referensi tertentu

### Keputusan teknis
- Metric utama: **AUC-PR** (bukan AUC-ROC, bukan accuracy) — karena lebih jujur untuk imbalanced dataset
- Split data: time-aware, 80% awal untuk training, 20% akhir untuk validasi — menghindari data leakage dari masa depan

---

## Step 2 — EDA & Data Quality

### Kenapa step ini penting
EDA bukan formalitas. Di sini kita cari sinyal: fitur mana yang punya pola berbeda antara fraud dan non-fraud.

### Yang ditemukan

**Pola waktu:**
Fraud lebih sering terjadi di jam malam (22.00–05.00) dan akhir pekan. Ini masuk akal secara bisnis — tim review lebih sedikit, dan pelaku fraud memanfaatkan momen tersebut.

**Pola amount:**
Distribusi TransactionAmt antara fraud dan non-fraud berbeda signifikan. Fraud cenderung clustering di nilai-nilai tertentu (bisa jadi testing amount sebelum fraud besar, atau pattern tertentu dari tool fraud automation).

**Fitur M (M1–M9):**
Kolom ini berisi flag T/F/NaN dari sistem Vesta — kemungkinan hasil verifikasi internal seperti "apakah alamat billing cocok", "apakah CVV valid", dll. Distribusinya berbeda nyata antara fraud dan non-fraud, jadi ini fitur penting.

**Missing values:**
Kolom-kolom dengan missing >90% kebanyakan dari identity table — wajar karena hanya 24% transaksi punya data identity. Strategi: drop kolom dengan missing >90% di training, impute sisanya dengan median.

### Keputusan teknis
EDA mengonfirmasi 3 kelompok fitur yang akan di-engineer:
1. Time features dari `TransactionDT`
2. M-flags sebagai integer encoded
3. Frequency encoding untuk categorical high-cardinality

---

## Step 3 — Feature Engineering

### Kenapa step ini penting
Raw data tidak langsung bisa dimakan model. Ada dua jenis transformasi yang perlu dilakukan: (1) bersihkan data agar model bisa membacanya, (2) tambah sinyal baru yang tidak ada di raw data.

### Pipeline yang dibangun

**1. Drop high-missing columns (threshold 90%)**
Kolom dengan >90% missing dihapus — terlalu banyak kosong untuk bisa diimputasi dengan bermakna.

**2. Time features dari `TransactionDT`**
```python
df['hour']        = (df['TransactionDT'] // 3600) % 24
df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7
df['is_night']    = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
```
`TransactionDT` mentah tidak informatif. Tapi setelah di-decompose, `is_night` dan `is_weekend` langsung punya korelasi dengan fraud.

**3. M-flag encoding**
```python
df[col] = df[col].map({'T': 1, 'F': 0}).fillna(-1).astype(int)
```
Kolom M1–M9 berisi string 'T'/'F'/NaN. Diubah ke integer: 1 = True, 0 = False, -1 = tidak diketahui. LightGBM bisa handle nilai -1 sebagai "unknown" yang berbeda dari 0.

**4. Frequency encoding untuk high-cardinality categoricals**
Kolom seperti `card1`, `card2`, `addr1` punya ribuan unique values. One-hot encoding tidak praktis. Solusinya: ganti setiap value dengan frekuensinya di training set.
```python
freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
df[f'{col}_freq'] = df[col].map(freq_maps[col]).fillna(0)
```
Nilai yang jarang muncul (mendekati 0) = lebih suspicious. Ini secara implisit menangkap sinyal "kartu baru yang tidak dikenal".

**5. Aggregation features**
```python
grp = df.groupby('card1')['TransactionAmt'].agg(['mean', 'std', 'count'])
```
Untuk setiap kartu (card1, card2) dan alamat (addr1), hitung mean/std/count TransactionAmt historis. Ini menangkap perilaku normal pengguna — transaksi yang jauh dari mean kartu tersebut lebih suspicious.

**6. has_identity flag**
```python
df['has_identity'] = df['id_01'].notna().astype(int)
```
Sederhana tapi penting: apakah transaksi ini punya data identity atau tidak.

**7. Median imputation**
Semua kolom numerik yang masih punya missing diisi dengan median dari training set.

### Keputusan kritis: artifact saving
Semua statistik yang dihitung dari training (freq_maps, agg_stats, medians) disimpan ke `fe_artifacts.pkl`. Ini penting untuk inference — model harus melihat data yang diproses dengan statistik yang **sama** dengan training. Kalau dihitung ulang dari data baru, hasilnya akan berbeda dan model akan error.

**Hasil:** 590.540 baris × 419 fitur, 0 missing values.

---

## Step 4 — Baseline Model

### Kenapa LightGBM
Untuk tabular fraud detection, LightGBM adalah pilihan yang solid karena:
- Handle missing values natively (tidak perlu semua kolom terisi)
- Categorical support built-in
- Efficient untuk dataset besar (472K training rows)
- Tidak perlu feature scaling

### Konfigurasi
```python
params = {
    "objective":    "binary",
    "metric":       "auc",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves":   63,
    "early_stopping_rounds": 50,
}
```
`early_stopping_rounds=50` memastikan training berhenti jika tidak ada improvement selama 50 rounds — mencegah overfitting.

### Hasil baseline
```
AUC-ROC  : 0.9609
AUC-PR   : 0.7951
```

AUC-ROC 0.96 artinya model sangat baik dalam ranking transaksi dari paling aman ke paling berisiko. AUC-PR 0.80 artinya bahkan di kelas minoritas (fraud), model masih cukup presisi.

### Feature importance
V-features (proprietary Vesta) mendominasi ~38% importance, diikuti C-features (~17%) dan aggregation features (~10%). Menariknya, aggregation features yang kita buat sendiri masuk top contributors — membuktikan feature engineering memberi nilai tambah nyata.

---

## Step 5 — Imbalanced Learning Experiments

### Masalah yang dihadapi
Dataset punya 3.5% fraud rate. Pertanyaannya: apakah perlu teknik khusus untuk menangani imbalance ini?

### Empat strategi yang diuji

**1. Baseline** — tidak ada perlakuan khusus, train langsung.

**2. Class Weighting** — beri bobot lebih tinggi pada kelas fraud saat training:
```python
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])  # ~27.6
```

**3. Random Undersampling** — buang sebagian data non-fraud agar ratio jadi 1:10.

**4. SMOTE** — buat synthetic samples fraud dengan interpolasi di feature space.

### Hasil perbandingan

| Strategy | AUC-ROC | AUC-PR | F1 | Recall |
|---|---|---|---|---|
| **Baseline** | **0.9609** | **0.7951** | 0.755 | 0.683 |
| Class Weight | 0.9642 | 0.763 | 0.720 | 0.660 |
| Undersampling | 0.9618 | 0.778 | 0.736 | 0.681 |
| SMOTE | 0.9576 | 0.779 | 0.738 | 0.684 |

### Insight penting
Baseline menang di AUC-PR — metric yang paling relevan. Ini counter-intuitive tapi ada penjelasannya:

- **LightGBM sudah robust terhadap imbalance** — algoritma boosting secara natural fokus ke hard examples di setiap iterasi
- **SMOTE di high-dimensional space berbahaya** — kita punya 419 fitur. Interpolasi linear di 419 dimensi menghasilkan synthetic samples yang tidak realistis. Misalnya, menginterpolasi antara dua transaksi fraud dari kota berbeda bisa menghasilkan transaksi dengan koordinat geografis yang tidak masuk akal
- **Undersampling membuang informasi** — dengan 472K training samples, setiap data berharga

**Keputusan:** Pakai baseline tanpa resampling.

---

## Step 6 — Evaluation & Risk Scoring

### Masalah threshold default

Model menghasilkan probabilitas (0–1). Untuk mengklasifikasikan "fraud/bukan fraud", perlu threshold. Default 0.5 adalah pilihan yang buruk untuk imbalanced data.

Dengan threshold 0.5 pada baseline:
```
Recall    : 36.4%  ← hanya 36% fraud yang tertangkap!
Precision : 86.6%
```

Artinya 64% fraud lolos. Ini tidak acceptable untuk sistem fraud detection.

### Threshold optimization

Threshold bukan hyperparameter model — ini keputusan bisnis yang dioptimasi terpisah. Dua pendekatan:

**F1-Maximization:** Cari threshold yang memaksimalkan F1-score.
```
Threshold: 0.246 → Recall: 68.3%, Precision: 85.2%, F1: 0.755
```

**Business Cost Minimization:** Definisi cost function:
```
total_cost = (FN_count × $500) + (FP_count × $10)
```
Scan semua threshold, pilih yang meminimalkan total cost.
```
Threshold: 0.093 → Recall: 80.1%, Precision: 55.3%, Cost: $0.21M
```

Dibandingkan tanpa model (approve semua):
```
Cost tanpa model: $0.64M
Cost dengan model: $0.21M
Savings: $0.43M
```

### Kenapa threshold 0.093 dipilih
Dengan cost FN ($500) jauh lebih besar dari FP ($10), model harus lebih "paranoid" — lebih baik false alarm daripada miss fraud. Threshold rendah (0.093) mencerminkan ini: transaksi dengan probabilitas fraud > 9.3% sudah di-flag.

### Risk bucket system
Selain binary classification, setiap transaksi diberi risk bucket untuk workflow yang berbeda:

| Bucket | Score | Action | Fraud Rate |
|---|---|---|---|
| Low | < 0.10 | Auto-approve | ~0.3% |
| Medium | 0.10 – 0.50 | Manual review | ~12% |
| High | ≥ 0.50 | Auto-block | ~55% |

Ini memungkinkan triase yang efisien: ~82% transaksi auto-approve tanpa review manusia, ~9% masuk antrian review, ~9% langsung diblok.

---

## Step 7 — API & Monitoring Dashboard

### Kenapa perlu API
Model yang hanya bisa dijalankan di notebook bukan solusi production. API memungkinkan sistem lain (payment gateway, mobile app, backend) memanggil model secara real-time.

### Arsitektur inference pipeline
Tantangan utama: pipeline feature engineering di inference harus **identik** dengan training. Semua statistik (freq_maps, agg_stats, medians) sudah disimpan di `fe_artifacts.pkl` saat training, dan di-load ulang saat API startup.

```
Request JSON
    ↓
Transaction schema validation (Pydantic)
    ↓
engineer_features():
    - time features
    - M-flag encoding
    - frequency encoding  ← pakai freq_maps dari training
    - aggregation features ← pakai agg_stats dari training
    - has_identity flag
    - median imputation   ← pakai medians dari training
    - align ke 419 feature cols
    ↓
model.predict_proba(X)
    ↓
risk_bucket + threshold check
    ↓
Response JSON
```

### Endpoints

**GET /health** — status model, threshold, metrics. Berguna untuk monitoring dan health check dari load balancer.

**POST /predict** — score satu transaksi. Query param `?explain=true` mengembalikan feature contributions (LightGBM built-in SHAP via `pred_contrib=True`).

**POST /predict/batch** — score hingga 1000 transaksi sekaligus. Lebih efisien karena preprocessing dilakukan sekali untuk seluruh batch.

### Explainability dengan pred_contrib
```python
contrib = model.predict(X, pred_contrib=True)[0]
# contrib[-1] = bias (expected value model)
# contrib[i]  = kontribusi fitur ke-i dalam log-odds unit
```
Ini SHAP-like values yang sudah built-in di LightGBM — tidak perlu library tambahan. Nilai positif artinya fitur tersebut mendorong prediksi ke arah fraud, negatif ke arah aman. Berguna untuk menjelaskan ke tim bisnis "kenapa transaksi ini di-flag".

### Latency
~37–65ms per prediction termasuk feature engineering lengkap. Cukup untuk real-time scoring di payment flow.

---

## Lessons Learned

### 1. Metric selection menentukan segalanya
AUC-PR lebih relevan dari AUC-ROC untuk imbalanced data. Kalau kita optimasi AUC-ROC, kita mungkin akan salah pilih strategi imbalanced learning (Class Weighting ROC-nya 0.9642 vs Baseline 0.9609 — tapi PR-nya lebih rendah).

### 2. Feature engineering > model complexity
Kita tidak melakukan hyperparameter tuning ekstensif. Dengan 419 fitur yang di-engineer dengan baik, baseline LightGBM sudah cukup baik. Waktu lebih baik diinvestasikan di feature engineering daripada di grid search.

### 3. Threshold adalah keputusan bisnis
Default threshold 0.5 membuat recall hanya 36%. Threshold yang dioptimasi dengan business cost meningkatkan recall ke 80% dengan savings $0.43M. Ini bukan tuning model — ini alignment antara model output dengan objective bisnis yang sebenarnya.

### 4. Inference pipeline harus production-ready sejak awal
Train-serve skew adalah salah satu penyebab paling umum model gagal di production. Dengan menyimpan semua FE artifacts dari awal, kita menghindari masalah ini.

### 5. Resampling tidak selalu membantu
SMOTE dan undersampling punya reputasi sebagai "solusi imbalanced data". Tapi pada dataset besar dengan model yang sudah robust seperti LightGBM, resampling justru bisa menambah noise. Selalu validasi dengan experiment yang terstruktur.

---

## Struktur Kode

```
src/
├── data/load_data.py          load_raw(), merge_tables(), load_processed()
├── features/build_features.py build_all_features() — training & inference
└── api/main.py                FastAPI: /health, /predict, /predict/batch

scripts/prepare_data.py        merge CSV → parquet (one-time)
tests/test_features.py         2 unit tests untuk FE pipeline
reports/dashboard.html         monitoring dashboard (buka di browser)
models/artifacts/              model + semua FE artifacts
```

### Dependensi kunci
- **pandas + pyarrow** — data wrangling + efficient storage (parquet)
- **lightgbm** — model utama
- **imbalanced-learn** — untuk experiment SMOTE/undersampling
- **fastapi + uvicorn** — serving layer
- **joblib** — serialization model dan artifacts
- **scikit-learn** — metrics, preprocessing utilities

---

## Cara Menjalankan

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Data pipeline (sekali saja)
python scripts/prepare_data.py

# Jalankan API
uvicorn src.api.main:app --reload

# Test API
curl -X POST http://localhost:8000/predict?explain=true \
  -H "Content-Type: application/json" \
  -d '{"TransactionDT": 86400, "TransactionAmt": 299.99, "ProductCD": "W", "card4": "visa"}'

# Dashboard
cd reports && python -m http.server 3000
# buka http://localhost:3000/dashboard.html

# Tests
python -m pytest tests/ -v
```
