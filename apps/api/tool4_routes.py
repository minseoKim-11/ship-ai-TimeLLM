from datetime import date
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from engine.tool4_buffett_score import compute_buffett_score_engine

router = APIRouter(
    prefix="/tool4",
    tags=["tool4"],
)


class BuffettScoreRequest(BaseModel):
    ticker: str  


@router.post("/buffett-score")
def post_buffett_score(body: BuffettScoreRequest) -> Dict[str, Any]:
    """
    Tool4: 워렌 버핏식 Quality + Valuation 점수 계산 (POST)
    - 입력: Ticker
    - 로직: 해당 Ticker의 가장 최신 재무 데이터를 기반으로 점수 산출
    """
    try:
        result = compute_buffett_score_engine(
            ticker=body.ticker
        )
    except ValueError as e:
        # 해당 티커의 데이터가 아예 없을 때 404반환
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"internal error: {e}")

    return result