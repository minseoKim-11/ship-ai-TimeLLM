import os
import pandas as pd

PROJECT_ROOT = "/workspace/ship-ai" 
MASTER_DENORM_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "master_table_denorm.csv")

# TOOL1의 합의점인 4가지 컬럼만 매핑
COLUMN_MAP = {
    "close":      "close",
    "ret_1d":     "ret_1d",      
    "roe":        "roe",
    "debt_ratio": "debt_ratio",
}

# 서버 시작 시 읽어두도록
_master_df = pd.read_csv(
    MASTER_DENORM_PATH,
    parse_dates=["date"],
    dtype={"ticker": str}
)
_master_df["ticker"] = _master_df["ticker"].astype(str).str.zfill(6)


def _series_to_list(df: pd.DataFrame, col: str):
    """
    df에서 ('date', col) 두 컬럼을 꺼내서:
      -> [["YYYY-MM-DD", value], ...] 형태로 변환
    """
    if col not in df.columns:
        return []

    tmp = (
        df[["date", col]]
        .dropna(subset=[col])
        .sort_values("date")
    )

    result = []
    for d, v in zip(tmp["date"], tmp[col]):
        result.append([d.strftime("%Y-%m-%d"), float(v)])
    return result


def get_stock_info_engine(
    ticker: str,
    start_date: str,
    end_date: str,
    channels: list[str] | None = None
) -> dict:
    """
    TOOL1 핵심 로직
    - 입력: ticker, start_date, end_date,
    - 출력: close / ret_1d / roe / debt_ratio 4개 시계열만 포함한 dict
    """

    ticker = str(ticker).zfill(6)
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)

    df_sub = _master_df[
        (_master_df["ticker"] == ticker) &
        (_master_df["date"] >= start) &
        (_master_df["date"] <= end)
    ].copy()

    # 데이터 전처리
    if df_sub.empty:
        return {
            "ticker": ticker,
            "date_range": {
                "start": start.strftime("%Y-%m-%d"),
                "end":   end.strftime("%Y-%m-%d")
            },
            "time_series": {},
            "meta": {
                "available_channels": [],
                "source": os.path.basename(MASTER_DENORM_PATH),
                "note": "no data in given range"
            }
        }
    if channels is None:
        target_json_keys = list(COLUMN_MAP.values())  
    else:
        target_json_keys = channels
        
    #  time_series dict 구성
    time_series = {}
    available_channels = []

    for raw_col, json_key in COLUMN_MAP.items():
        if json_key not in target_json_keys:
            continue
        if raw_col not in df_sub.columns:
            continue

        arr = _series_to_list(df_sub, raw_col)
        time_series[json_key] = arr
        if len(arr) > 0:
            available_channels.append(json_key)

    # 최종 응답 dict
    response = {
    "ticker": ticker,
    "date_range": {
        "start": start.strftime("%Y-%m-%d"),
        "end":   end.strftime("%Y-%m-%d")
    },
    "time_series": time_series,
    "meta": {
        "channels": sorted(time_series.keys()), 
        "source": os.path.basename(MASTER_DENORM_PATH)
    }
}
    return response