import os
import math
import numpy as np
import pandas as pd
import torch
from typing import Optional

from engine.time_llm_config import load_time_llm_model, Configs
from engine.data_handler import DataHandler  

# --------------------------
# 공통 상수 & 경로
# --------------------------
PROJECT_ROOT = "/workspace/ship-ai"

MASTER_SCALED_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "final_master_table_v2.csv"
)

MASTER_DENORM_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "master_table_denorm.csv"
)

TAU_THRESHOLD_SCALED = -0.0975  
INPUT_SEQ_LEN = 120
PRED_LEN = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 전역 객체 초기화
# --------------------------

# (1) DataHandler
data_handler = DataHandler(
    MASTER_SCALED_PATH,
    train_end_date="2022-12-31"
)

# (2) denorm master table
master_denorm_df = pd.read_csv(
    MASTER_DENORM_PATH,
    parse_dates=["date"],
    dtype={"ticker": str},
)

# (3) Time-LLM 모델 (time_llm_config 에서만 로딩)
model, configs = load_time_llm_model(DEVICE)
model.eval()

print(f"[TOOL3] Loaded Time-LLM on {DEVICE}")

# ==========================
# 유틸 함수들
# ==========================

def _get_latest_as_of_date(ticker: str) -> pd.Timestamp:
    """해당 티커의 가장 최신 date 반환"""
    df_t = master_denorm_df[master_denorm_df["ticker"] == ticker]
    if df_t.empty:
        raise ValueError(f"[TOOL3] No denorm data for ticker={ticker}")
    return df_t["date"].max()


def _get_price_window_for_vol(
    ticker: str,
    as_of: pd.Timestamp,
    lookback_days: int = 60
) -> pd.Series:
    """
    변동성 계산용으로 as_of 기준 과거 lookback_days+1개 close 가격 시계열 반환
    (denorm 공간)
    """
    df_t = (
        master_denorm_df[master_denorm_df["ticker"] == ticker]
        .sort_values("date")
    )

    df_t = df_t[df_t["date"] <= as_of]
    if len(df_t) < lookback_days + 2:
        # 데이터 적으면 있는 만큼만
        return df_t["close"]

    return df_t.iloc[-(lookback_days + 1):]["close"]


def _compute_realized_vol_from_close(close_series: pd.Series) -> float:
    """
    로그수익률 기반 연율 변동성이나 단순 std를 계산.
    여기서는 단순하게:
    - log_return = log(P_t) - log(P_{t-1})
    - realized_vol = std(log_return)
    """
    values = close_series.values.astype(float)
    if len(values) < 2:
        return 0.0

    log_price = np.log(values)
    log_ret = np.diff(log_price)  # (N-1,)

    if log_ret.std() < 1e-8:
        return 0.0

    # 연율화
    vol = log_ret.std()
    return float(vol)


def _inverse_close_from_scaled(
    scaled_vec: np.ndarray,
    ticker: str,
    data_handler: DataHandler,
) -> np.ndarray:
    """
    scaler가 티커별로 data_handler.scalers_by_ticker[ticker] 에 저장되어 있다고 가정.
    scaled_vec: (T,) 형태 (close 채널 스케일된 값들)
    return: (T,) 형태의 복원된 close_log 값들
    """
    # 티커 포맷 통일 (이미 밖에서 zfill(6) 했으면 그대로 들어옴)
    ticker = str(ticker).zfill(6)

    if ticker not in data_handler.scalers_by_ticker:
        raise ValueError(
            f"[TOOL3] No scaler found for ticker={ticker} in data_handler.scalers_by_ticker"
        )

    scaler = data_handler.scalers_by_ticker[ticker]  # ★ 여기 핵심

    # scaler가 StandardScaler 라고 가정
    C = scaler.mean_.shape[0]     # 전체 채널 수

    tmp = np.zeros((len(scaled_vec), C), dtype=np.float32)
    tmp[:, 0] = scaled_vec  # 0번 채널이 close_log 스케일이라고 가정

    inv = scaler.inverse_transform(tmp)
    return inv[:, 0]  # denorm된 close_log


def _build_input_window_scaled(
    ticker: str
) -> torch.Tensor:
    """
    DataHandler에서 티커별 전체 스케일된 데이터를 가져와
    최신 기준으로 마지막 INPUT_SEQ_LEN만 잘라서 모델 입력 텐서를 만든다.

    return: (1, Seq, C) 텐서
    """
    df_scaled = data_handler.get_scaled_data_by_ticker(ticker)

    if df_scaled is None or len(df_scaled) < INPUT_SEQ_LEN:
        raise ValueError(
            f"[TOOL3] Not enough scaled data for ticker={ticker}. "
            f"Need at least {INPUT_SEQ_LEN} rows."
        )

    # numpy array로 변환 (T, C)
    if hasattr(df_scaled, "values"):
        x_all = df_scaled.values.astype(np.float32)
    else:
        x_all = np.asarray(df_scaled, dtype=np.float32)

    x_win = x_all[-INPUT_SEQ_LEN:, :]  # (Seq, C)
    x_win = x_win[None, :, :]          # (1, Seq, C)
    x_tensor = torch.from_numpy(x_win).to(DEVICE)

    return x_tensor


# ==========================
# 메인 엔진 함수
# ==========================

def get_forecast_gap_engine(
    ticker: str,
    as_of_date: Optional[str] = None,
    horizon_days: int = PRED_LEN,
    tau_threshold_scaled: float = TAU_THRESHOLD_SCALED,
) -> dict:
    """
    TOOL3 엔진 핵심:
    - 입력: ticker, (옵션) as_of_date, (옵션) horizon_days(현재 10 고정)
    - 출력: forecast + gap 분석 JSON (엔진용)
    """

    # -------- 1) 입력 정리 --------
    ticker = str(ticker).zfill(6)

    if as_of_date is None:
        as_of = _get_latest_as_of_date(ticker)
    else:
        as_of = pd.to_datetime(as_of_date)

    if horizon_days != PRED_LEN:
        # V1에선 10일 예측 모델만 있으므로 일단 warning 수준
        print(
            f"[TOOL3] WARNING: horizon_days={horizon_days}, "
            f"but model is trained with PRED_LEN={PRED_LEN}. "
            f"V1에서는 {PRED_LEN}으로 고정 동작합니다."
        )
        horizon_days = PRED_LEN

    # -------- 2) 모델 입력 생성 (scaled space) --------
    x_scaled = _build_input_window_scaled(ticker)  # (1, Seq, C)
    B, Seq, C = x_scaled.shape

    dummy_mark_enc = torch.zeros(B, Seq, 4, device=DEVICE)
    dummy_mark_dec = torch.zeros(B, horizon_days, 4, device=DEVICE)
    dummy_dec_in   = torch.zeros(B, horizon_days, C, device=DEVICE)

    # -------- 3) 모델 예측 --------
    model.eval()
    with torch.no_grad():
        outputs = model(x_scaled, dummy_mark_enc, dummy_dec_in, dummy_mark_dec)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        preds_scaled = outputs[:, -horizon_days:, :]  # (1, Pred, C)

    # close_log 채널 (scaled space)
    pred_close_scaled = preds_scaled[0, :, 0].detach().cpu().numpy()   # (Pred,)
    last_close_scaled = x_scaled[0, -1, 0].detach().cpu().item()

    # -------- 4) scaled → close_log 복원 → 가격으로 변환 --------
    # (1) 현재/예측 close_log 복원
    close_log_vec = _inverse_close_from_scaled(
        np.concatenate([[last_close_scaled], pred_close_scaled]),
        ticker,        # ← 두 번째 인자: ticker
        data_handler,  # ← 세 번째 인자: data_handler
    )  # shape: (Pred+1,)

    current_close_log = close_log_vec[0]
    pred_close_log_vec = close_log_vec[1:]  # (Pred,)

    # (2) 로그 → 가격
    current_price = float(math.exp(current_close_log))
    pred_price_path = np.exp(pred_close_log_vec)  # (Pred,)

    # -------- 5) 누적 수익률 / 변동성 / gap 계산 --------
    # 누적 로그수익률 = 마지막 log - 시작 log
    pred_cum_return_log = float(pred_close_log_vec[-1] - current_close_log)

    # 단순 수익률
    pred_cum_return = float((pred_price_path[-1] - current_price) / current_price)
    pred_cum_return_pct = pred_cum_return * 100.0

    # 변동성: denorm close 기반
    close_window = _get_price_window_for_vol(ticker, as_of, lookback_days=60)
    realized_vol = _compute_realized_vol_from_close(close_window)

    if realized_vol > 1e-8:
        gap_score = pred_cum_return_log / realized_vol
    else:
        gap_score = 0.0

    # -------- 6) 방향 레이블 (τ* 기준) --------
    if pred_cum_return_log >= tau_threshold_scaled:
        direction_label = "UP_or_NEUTRAL"
    else:
        direction_label = "DOWN"

    # 간단한 confidence: gap_score 절대값을 0~1로 squash
    direction_confidence = float(
        0.5 + 0.5 * math.tanh(abs(gap_score))
    )  # 대략 0.5 ~ 1.0

    # -------- 7) 날짜 path 구성 (영업일 기준 간단 버전) --------
    # denorm master에서 as_of 이후 horizon_days 개수만큼 날짜를 가져오거나,
    # 없으면 단순히 +1day씩 늘려도 됨. 여기서는 간단히 +1day 버전.
    dates = []
    cur_date = as_of
    for i in range(1, horizon_days + 1):
        cur_date = cur_date + pd.Timedelta(days=1)
        dates.append(cur_date.strftime("%Y-%m-%d"))

    predicted_price_path = [
        [d, float(p)] for d, p in zip(dates, pred_price_path)
    ]

    # -------- 8) 최종 응답 JSON --------
    response = {
        "ticker": ticker,
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "pred_horizon_days": horizon_days,
        "tau_threshold_scaled": tau_threshold_scaled,

        "scaled_space": {
            "current_close_log": float(current_close_log),
            "pred_close_log_path": pred_close_log_vec.tolist(),
            "pred_cum_return_log": float(pred_cum_return_log),
            "realized_vol_lookback_60d": float(realized_vol),
        },

        "denorm": {
            "current_price": float(current_price),
            "predicted_price_path": predicted_price_path,
            "pred_cum_return": float(pred_cum_return),
            "pred_cum_return_pct": float(pred_cum_return_pct),
            "gap_score": float(gap_score),
            "direction_label": direction_label,
            "direction_confidence": float(direction_confidence),
        }
    }

    return response
