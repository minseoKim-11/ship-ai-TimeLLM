from fastapi import APIRouter
from engine.tool1_stock_info import get_stock_info_engine

router = APIRouter()

@router.get("/tool1/stock-info")
def tool1_stock_info(ticker: str, start_date: str, end_date: str):
    engine_resp = get_stock_info_engine(ticker, start_date, end_date)
    return {
        "status": "ok",
        "response": engine_resp
    }