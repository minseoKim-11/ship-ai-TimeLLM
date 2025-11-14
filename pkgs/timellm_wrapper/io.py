import pandas as pd
from pathlib import Path
import numpy as np

# 일단 CSV/Parquet 어느쪽이든 읽기 시도
DATA_DIR = Path("/workspace/ship-ai/data/silver")  # 필요 시 경로 바꿔도 됨

def load_minimal_df(ticker: str) -> pd.DataFrame:
    """prices_daily/{ticker}.parquet 또는 csv에서 일봉 로드 (date, close 필수)"""
    p_parq = DATA_DIR / "prices_daily" / f"{ticker}.parquet"
    p_csv  = DATA_DIR / "prices_daily" / f"{ticker}.csv"
    if p_parq.exists():
        df = pd.read_parquet(p_parq)
    elif p_csv.exists():
        df = pd.read_csv(p_csv)
    else:
        raise FileNotFoundError(f"prices_daily not found for {ticker} under {p_parq} or {p_csv}")

    # 최소 컬럼 체크
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError("prices_daily requires columns: date, close")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def to_patches(arr: np.ndarray, patch_len=8, stride=8):
    """
    arr: [T, C] → patches: [P, patch_len*C]
    Time-LLM 실제 연결 전까지는 형태만 맞추는 유틸.
    """
    T, C = arr.shape
    patches = []
    for s in range(0, T - patch_len + 1, stride):
        seg = arr[s:s+patch_len].reshape(-1)
        patches.append(seg)
    return np.stack(patches) if patches else np.empty((0, patch_len*C), dtype=arr.dtype)
