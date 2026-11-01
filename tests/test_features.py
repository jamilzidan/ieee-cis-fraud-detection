import pandas as pd
import numpy as np
from src.features.build_features import add_time_features, add_frequency_encoding


def make_dummy_df(n=100):
    return pd.DataFrame({
        "TransactionID":  range(n),
        "TransactionDT":  np.random.randint(0, 86400 * 30, n),
        "TransactionAmt": np.random.exponential(100, n),
        "ProductCD":      np.random.choice(["W", "H", "C", "S", "R"], n),
        "card1":          np.random.randint(1000, 9999, n),
        "card2":          np.random.randint(100, 999, n),
        "P_emaildomain":  np.random.choice(["gmail.com", "yahoo.com", "hotmail.com"], n),
        "isFraud":        np.random.choice([0, 1], n, p=[0.965, 0.035]),
    })


def test_time_features():
    df = make_dummy_df()
    out = add_time_features(df)
    assert "hour" in out.columns
    assert "day_of_week" in out.columns
    assert out["hour"].between(0, 23).all()


def test_frequency_encoding():
    df = make_dummy_df()
    out, freq_maps = add_frequency_encoding(df, ["ProductCD"])
    assert "ProductCD_freq" in out.columns
    assert out["ProductCD_freq"].between(0, 1).all()
    assert "ProductCD" in freq_maps
