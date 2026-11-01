"""Script: load raw CSVs, merge, dan simpan sebagai parquet."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.load_data import load_raw, merge_tables
from pathlib import Path

OUTPUT = Path("data/processed/train_merged.parquet")

if __name__ == "__main__":
    transaction, identity = load_raw()
    df = merge_tables(transaction, identity)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    print(f"Saved {len(df):,} rows → {OUTPUT}")
