from fastapi import APIRouter
from engine.tool2_macro_pulse import get_macro_pulse_engine
from .tool2_schemas import Tool2MacroPulseRequest

router = APIRouter(prefix="/tool2", tags=["Tool2"])

@router.post("/macro")
def macro_pulse(body: Tool2MacroPulseRequest):
    """
    Tool2: 조선업 업황 및 지표 변화률 분석 (POST)
    - 입력: null
    - 로직: 조선업의 시황을 스냅샷으로 제공함
    """
    return get_macro_pulse_engine()
