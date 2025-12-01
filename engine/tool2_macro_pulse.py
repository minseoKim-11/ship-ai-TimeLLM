import os
import pandas as pd
import numpy as np

PROJECT_ROOT = "/workspace/ship-ai" 
MASTER_DENORM_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "master_table_denorm.csv")

_raw_df = pd.read_csv(
    MASTER_DENORM_PATH,
    parse_dates=["date"],
    dtype={"ticker": str}
)

# 조선업 업황은 "ticker"와 무관한 전역 시계열이므로 날짜별로 한 줄만 남기도록 정규화
# 여러 ticker가 같은 macro 값을 중복 갖고 있다면 첫 번째만 사용)
MACRO_COLS = ["bdi_proxy", "newbuild_proxy", "wti"]
cols_exist = [c for c in MACRO_COLS if c in _raw_df.columns]

_macro_df = (
    _raw_df[["date"] + cols_exist]
    .drop_duplicates(subset=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)

# 유틸 함수
def _compute_indicator_stats(df: pd.DataFrame, col: str, as_of_date: pd.Timestamp, window_days: int):
    """
    주어진 indicator(col)에 대해
    - as_of_date 기준 window_days 일 동안의 구간을 잡고
    - level(현재 값), mom(구간 첫 값 대비 변화율),
      mean, std, zscore(현재 값의 z-score)를 계산한다.
    데이터가 부족하면 None 반환.
    """
    if col not in df.columns:
        return None

    end = as_of_date
    start = as_of_date - pd.Timedelta(days=window_days)

    sub = df[(df["date"] >= start) & (df["date"] <= end)][["date", col]].dropna(subset=[col])

    if len(sub) < 5:
        # 데이터가 너무 적으면 스킵
        return None

    sub = sub.sort_values("date")

    series = sub[col].astype(float)

    level = float(series.iloc[-1])
    first = float(series.iloc[0])

    # 첫 값 기준 변화율 (모멘텀)
    if abs(first) > 1e-8:
        mom = (level - first) / abs(first)   # 예: 0.05 → +5%
    else:
        mom = 0.0

    mean = float(series.mean())
    std = float(series.std(ddof=0))  # 모집단 표준편차

    if std > 1e-8:
        zscore = (level - mean) / std
    else:
        zscore = 0.0

    return {
        "level": level,
        "mom": mom,
        "mean": mean,
        "std": std,
        "zscore": zscore,
    }

# PUBLIC: Tool2 엔진 진입 함수
def get_macro_pulse_engine() -> dict:
    """
    TOOL2: 조선업 업황 스냅샷 (Macro Pulse)
    - 입력 파라미터 없음
    - master_table_denorm.csv 전체를 보고, 가장 최신 날짜(as_of_date)를 찾는다.
    - as_of_date 기준으로 90일/365일 업황 지표 요약을 반환한다.
    """

    if _macro_df.empty:
        return {
            "as_of_date": None,
            "windows": {},
            "meta": {
                "indicator_keys": [],
                "source": os.path.basename(MASTER_DENORM_PATH),
                "note": "no macro data available"
            }
        }

    as_of_date = _macro_df["date"].max()

    indicators = [c for c in MACRO_COLS if c in _macro_df.columns]

    # 2단계 윈도우 설정 
    # TODO: 추후 고도화
    windows_spec = {
        "short_90d": 90,
        "mid_365d": 365,
    }

    windows = {}

    for win_key, days in windows_spec.items():
        indicators_stats = {}

        for col in indicators:
            stats = _compute_indicator_stats(_macro_df, col, as_of_date, days)
            if stats is not None:
                indicators_stats[col] = stats

        windows[win_key] = {
            "days": days,
            "indicators": indicators_stats
        }

    response = {
        "as_of_date": as_of_date.strftime("%Y-%m-%d"),
        "windows": windows,
        "meta": {
            "indicator_keys": indicators,
            "source": os.path.basename(MASTER_DENORM_PATH)
        }
    }
    return response
