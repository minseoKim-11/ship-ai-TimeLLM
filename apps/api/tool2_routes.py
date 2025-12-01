from fastapi import APIRouter
from engine.tool2_macro_pulse import get_macro_pulse_engine

router = APIRouter()

@router.get("/tool2/macro-pulse")
def macro_pulse():
    """
    TOOL2: 조선업 업황 스냅샷
    - 입력 파라미터 없음
    """
    return get_macro_pulse_engine()
