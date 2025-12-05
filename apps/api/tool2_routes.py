from fastapi import APIRouter
from engine.tool2_macro_pulse import get_macro_pulse_engine
from .tool2_schemas import Tool2MacroPulseRequest

router = APIRouter(prefix="/tool2", tags=["Tool2"])

@router.post("/macro")
def macro_pulse(body: Tool2MacroPulseRequest):
    """
    TOOL2: 조선업 업황 스냅샷
    - 입력 파라미터는 현재 없음
    """
    return get_macro_pulse_engine()
