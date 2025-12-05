from typing import List, Optional
from pydantic import BaseModel

# 개별 (날짜, 가격) 포인트
class PricePoint(BaseModel):
    date: str
    price: float

# scaled space (로그/스케일 공간 정보)
class Tool3ScaledSpace(BaseModel):
    current_close_log: float
    pred_close_log_path: List[float]
    pred_cum_return_log: float
    realized_vol_lookback_60d: float

# 실제 가격 공간
class Tool3Denorm(BaseModel):
    current_price: float
    predicted_price_path: List[PricePoint]
    pred_cum_return: float
    pred_cum_return_pct: float
    gap_score: float
    direction_label: str
    direction_confidence: float

# 요청 바디
# 요청 바디
# 요청 바디
class Tool3ForecastRequest(BaseModel):
    """
    - as_of_date, horizon_days 는 서버 내부에서 결정
    """
    ticker: str


# 응답 바디
class Tool3ForecastResponse(BaseModel):
    ticker: str
    as_of_date: str
    pred_horizon_days: int
    tau_threshold_scaled: float
    scaled_space: Tool3ScaledSpace
    denorm: Tool3Denorm
