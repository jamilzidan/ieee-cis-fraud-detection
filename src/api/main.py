from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import json
import time
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path(__file__).parents[2] / "models" / "artifacts"
MODEL_PATH    = ARTIFACTS_DIR / "lgbm_baseline.pkl"
FE_PATH       = ARTIFACTS_DIR / "fe_artifacts.pkl"
FEAT_PATH     = ARTIFACTS_DIR / "feature_cols.json"
CARD_PATH     = ARTIFACTS_DIR / "model_card.json"
THR_PATH      = ARTIFACTS_DIR / "best_threshold.json"

# ── app ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    description="Real-time fraud scoring untuk transaksi e-commerce.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── globals (loaded at startup) ────────────────────────────────────────────
model        = None
fe_artifacts = None
feature_cols = None
threshold    = 0.5
model_card   = None


@app.on_event("startup")
def load_artifacts():
    global model, fe_artifacts, feature_cols, threshold, model_card
    model        = joblib.load(MODEL_PATH)
    fe_artifacts = joblib.load(FE_PATH)
    with open(FEAT_PATH)  as f: feature_cols = json.load(f)
    with open(CARD_PATH)  as f: model_card   = json.load(f)
    with open(THR_PATH)   as f: thr_data     = json.load(f)
    threshold = thr_data.get("threshold", 0.5)
    print(f"[startup] model loaded | threshold={threshold} | features={len(feature_cols)}")


# ── schemas ────────────────────────────────────────────────────────────────
class Transaction(BaseModel):
    TransactionDT:   int
    TransactionAmt:  float
    ProductCD:       Optional[str]   = None
    card1:           Optional[float] = None
    card2:           Optional[float] = None
    card3:           Optional[float] = None
    card4:           Optional[str]   = None
    card5:           Optional[float] = None
    card6:           Optional[str]   = None
    addr1:           Optional[float] = None
    addr2:           Optional[float] = None
    dist1:           Optional[float] = None
    dist2:           Optional[float] = None
    P_emaildomain:   Optional[str]   = None
    R_emaildomain:   Optional[str]   = None
    DeviceType:      Optional[str]   = None


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="0–1 score, semakin tinggi semakin berisiko")
    risk_bucket:       str   = Field(..., description="Low | Medium | High")
    is_fraud:          bool  = Field(..., description="True jika score >= threshold")
    threshold_used:    float
    latency_ms:        float


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    threshold:    float
    n_features:   int
    model_metrics: dict


# ── feature engineering (inference) ───────────────────────────────────────
def engineer_features(tx: Transaction) -> pd.DataFrame:
    row = tx.model_dump()
    df  = pd.DataFrame([row])

    fe = fe_artifacts

    # 1. Time features
    df["hour"]        = (df["TransactionDT"] // 3600) % 24
    df["day_of_week"] = (df["TransactionDT"] // (3600 * 24)) % 7
    df["is_night"]    = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # 2. M-features (not in transaction schema, fill -1)
    for col in fe["m_cols"]:
        if col not in df.columns:
            df[col] = -1

    # 3. Frequency encoding
    for col in fe["freq_cols"]:
        fmap = fe["freq_maps"].get(col, {})
        if col in df.columns:
            df[f"{col}_freq"] = df[col].map(fmap).fillna(0)
            df = df.drop(columns=[col])
        else:
            df[f"{col}_freq"] = 0.0

    # 4. Aggregation features
    for g_col, stats_dict in fe["agg_stats"].items():
        if g_col in df.columns:
            val = df[g_col].iloc[0]
            for stat_col, mapping in stats_dict.items():
                df[stat_col] = mapping.get(val, np.nan)
        else:
            for stat_col in stats_dict:
                df[stat_col] = np.nan

    # 5. has_identity (no identity cols in input → 0)
    df["has_identity"] = 0

    # 6. Drop leftover object cols
    obj_cols = df.select_dtypes("object").columns.tolist()
    df = df.drop(columns=obj_cols, errors="ignore")

    # 7. Median impute
    for col, med in fe["medians"].items():
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(med)

    # 8. Align to feature_cols (fill missing with 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df[feature_cols]


def risk_bucket(prob: float) -> str:
    if prob < 0.10:
        return "Low"
    if prob < 0.50:
        return "Medium"
    return "High"


# ── endpoints ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        threshold=threshold,
        n_features=len(feature_cols) if feature_cols else 0,
        model_metrics=model_card.get("metrics", {}) if model_card else {},
    )


@app.post("/predict", tags=["scoring"])
def predict(tx: Transaction, explain: bool = False):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    try:
        X    = engineer_features(tx)
        prob = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering error: {e}")

    latency = (time.perf_counter() - t0) * 1000

    resp = PredictionResponse(
        fraud_probability=round(prob, 6),
        risk_bucket=risk_bucket(prob),
        is_fraud=prob >= threshold,
        threshold_used=threshold,
        latency_ms=round(latency, 2),
    )

    if not explain:
        return resp

    # LightGBM built-in SHAP — pred_contrib returns shape (n, n_features+1)
    # last column is expected value (bias); rest are per-feature contributions
    try:
        contrib = model.predict(X, pred_contrib=True)[0]
        bias    = float(contrib[-1])
        contribs = {col: float(contrib[i]) for i, col in enumerate(feature_cols)}
        top_pos  = sorted(contribs.items(), key=lambda x: x[1],  reverse=True)[:5]
        top_neg  = sorted(contribs.items(), key=lambda x: x[1])[:5]
        def safe_float(x):
            v = float(x)
            return None if (v != v or abs(v) == float("inf")) else round(v, 4)

        top_features = [
            {"feature": k, "contribution": round(v, 4), "value": safe_float(X[k].iloc[0])}
            for k, v in top_pos + top_neg if abs(v) > 1e-6
        ]
    except Exception as ex:
        print(f"[explain error] {ex}")
        top_features, bias = [], 0.0

    return {
        **resp.model_dump(),
        "explanation": {
            "bias": round(bias, 4),
            "top_features": top_features,
            "note": "contribution = log-odds unit (positive = push toward fraud)",
        },
    }


@app.post("/predict/batch", tags=["scoring"])
def predict_batch(transactions: list[Transaction]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(transactions) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 transaksi per batch")

    t0 = time.perf_counter()
    frames = [engineer_features(tx) for tx in transactions]
    X      = pd.concat(frames, ignore_index=True)
    probs  = model.predict_proba(X)[:, 1]
    latency = (time.perf_counter() - t0) * 1000

    return {
        "count": len(transactions),
        "latency_ms": round(latency, 2),
        "results": [
            {
                "fraud_probability": round(float(p), 6),
                "risk_bucket": risk_bucket(float(p)),
                "is_fraud": float(p) >= threshold,
            }
            for p in probs
        ],
    }
