from fastapi import APIRouter, HTTPException
from engine.tool3_forecast_gap import get_forecast_gap_engine
from .tool3_schemas import (
    Tool3ForecastRequest,
    Tool3ForecastResponse,
    Tool3ScaledSpace,
    Tool3Denorm,
    PricePoint,
)

router = APIRouter(
    prefix="/tool3",
    tags=["Tool3"],
)

@router.post("/forecast-gap", response_model=Tool3ForecastResponse)
def tool3_forecast_gap(body: Tool3ForecastRequest):
    """
    Time-LLM 기반 TOOL3 갭 분석 API
    """
    try:
        result = get_forecast_gap_engine(
            ticker=body.ticker,
            as_of_date=body.as_of_date,
            horizon_days=body.horizon_days,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    # result(dict)를 Pydantic 모델로 변환
    scaled = result["scaled_space"]
    denorm = result["denorm"]

    resp = Tool3ForecastResponse(
        ticker=result["ticker"],
        as_of_date=result["as_of_date"],
        pred_horizon_days=result["pred_horizon_days"],
        tau_threshold_scaled=result["tau_threshold_scaled"],
        scaled_space=Tool3ScaledSpace(
            current_close_log=scaled["current_close_log"],
            pred_close_log_path=scaled["pred_close_log_path"],
            pred_cum_return_log=scaled["pred_cum_return_log"],
            realized_vol_lookback_60d=scaled["realized_vol_lookback_60d"],
        ),
        denorm=Tool3Denorm(
            current_price=denorm["current_price"],
            predicted_price_path=[
                PricePoint(date=d, price=p)
                for d, p in denorm["predicted_price_path"]
            ],
            pred_cum_return=denorm["pred_cum_return"],
            pred_cum_return_pct=denorm["pred_cum_return_pct"],
            gap_score=denorm["gap_score"],
            direction_label=denorm["direction_label"],
            direction_confidence=denorm["direction_confidence"],
        ),
    )

    return resp
