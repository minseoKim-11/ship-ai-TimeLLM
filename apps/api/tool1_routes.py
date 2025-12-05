from fastapi import APIRouter
from engine.tool1_stock_info import get_stock_info_engine
from .tool1_schemas import Tool1StockInfoRequest

router = APIRouter(prefix="/tool1", tags=["Tool1"])

@router.post("/stock-info")
def tool1_stock_info(body: Tool1StockInfoRequest):
    """
    Tool1: 기업 재무 조회 (POST)
    - 입력: ticker, start_date, end_date
    - 로직: 기업의 종가, 일일수익률, roe, 부채비율을 제공함
    """
    engine_resp = get_stock_info_engine(
        ticker=body.ticker,
        start_date=body.start_date,
        end_date=body.end_date,
    )
    return {
        "status": "ok",
        "response": engine_resp,
    }
