import pandas as pd
import numpy as np
from typing import Optional


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()
    cols_to_drop = [c for c in cols_to_drop if c not in ['TransactionID', 'isFraud']]
    return df.drop(columns=cols_to_drop)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['hour']        = (df['TransactionDT'] // 3600) % 24
    df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7
    df['is_night']    = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    return df


def encode_m_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m_cols = [c for c in df.columns if c.startswith('M') and df[c].dtype == 'object']
    for col in m_cols:
        df[col] = df[col].map({'T': 1, 'F': 0}).fillna(-1).astype(int)
    return df


def add_frequency_encoding(
    df: pd.DataFrame,
    cols: Optional[list] = None,
    freq_maps: Optional[dict] = None,
) -> tuple:
    """
    Encode kategorikal dengan frekuensi.
    Jika freq_maps diberikan (saat inference), pakai maps itu.
    Jika tidak, hitung dari df (saat training).
    Return (df_transformed, freq_maps) supaya maps bisa disimpan dan dipakai ulang.
    """
    df = df.copy()
    if cols is None:
        cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'DeviceType']
    cols = [c for c in cols if c in df.columns]

    if freq_maps is None:
        freq_maps = {}
        for col in cols:
            freq_maps[col] = df[col].value_counts(normalize=True, dropna=False).to_dict()

    for col in cols:
        df[f'{col}_freq'] = df[col].map(freq_maps.get(col, {})).fillna(0)
        df = df.drop(columns=[col])

    return df, freq_maps


def add_aggregation_features(df: pd.DataFrame, agg_stats: Optional[dict] = None) -> tuple:
    """
    Tambah aggregasi per entitas.
    Jika agg_stats diberikan (saat inference), pakai stats itu.
    Return (df_transformed, agg_stats).
    """
    df = df.copy()
    agg_groups = {
        'card1': ['TransactionAmt'],
        'card2': ['TransactionAmt'],
        'addr1': ['TransactionAmt'],
    }

    if agg_stats is None:
        agg_stats = {}
        for group_col, value_cols in agg_groups.items():
            if group_col not in df.columns:
                continue
            for val_col in value_cols:
                if val_col not in df.columns:
                    continue
                grp = df.groupby(group_col)[val_col].agg(['mean', 'std', 'count'])
                grp.columns = [f'{group_col}_{val_col}_{s}' for s in ['mean', 'std', 'count']]
                agg_stats[group_col] = grp

    for group_col, grp in agg_stats.items():
        if group_col in df.columns:
            df = df.join(grp, on=group_col)

    return df, agg_stats


def add_identity_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'id_01' in df.columns:
        df['has_identity'] = df['id_01'].notna().astype(int)
    return df


def impute_numeric(df: pd.DataFrame, medians: Optional[dict] = None) -> tuple:
    df = df.copy()
    # Drop sisa kolom object
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    obj_cols = [c for c in obj_cols if c not in ['isFraud']]
    df = df.drop(columns=obj_cols)

    num_cols = [c for c in df.columns
                if df[c].isnull().any() and c not in ['TransactionID', 'isFraud']]

    if medians is None:
        medians = df[num_cols].median().to_dict()

    df[num_cols] = df[num_cols].fillna(medians)
    return df, medians


def build_all_features(df: pd.DataFrame, artifacts: Optional[dict] = None) -> tuple:
    """
    Full pipeline. artifacts berisi freq_maps, agg_stats, medians dari training.
    Kalau None, hitung semuanya (mode training).
    Return (df_features, artifacts).
    """
    if artifacts is None:
        artifacts = {}

    df = drop_high_missing(df)
    df = add_time_features(df)
    df = encode_m_features(df)
    df = add_identity_flag(df)

    df, freq_maps = add_frequency_encoding(df, freq_maps=artifacts.get('freq_maps'))
    artifacts['freq_maps'] = freq_maps

    df, agg_stats = add_aggregation_features(df, agg_stats=artifacts.get('agg_stats'))
    artifacts['agg_stats'] = agg_stats

    df, medians = impute_numeric(df, medians=artifacts.get('medians'))
    artifacts['medians'] = medians

    return df, artifacts
