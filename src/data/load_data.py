import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / "data"


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    transaction = pd.read_csv(DATA_DIR / "raw" / "train_transaction.csv")
    identity    = pd.read_csv(DATA_DIR / "raw" / "train_identity.csv")
    return transaction, identity


def merge_tables(transaction: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    return transaction.merge(identity, on="TransactionID", how="left")


def load_processed() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "processed" / "train_merged.parquet")
