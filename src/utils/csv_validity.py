from pathlib import Path

import pandas as pd


def csv_validity(path: Path, min_rows: int = 1):
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        return len(pd.read_csv(path, nrows=min_rows)) >= min_rows
    except Exception:
        return False
