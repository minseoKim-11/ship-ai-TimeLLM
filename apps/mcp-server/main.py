from fastapi import FastAPI
from pydantic import BaseModel
from timellm_wrapper import TimeLLMWrapper

app = FastAPI()
timellm = TimeLLMWrapper()

class DislocationReq(BaseModel):
    companyName: str
    analysisDate: str  # YYYY-MM-DD

@app.post("/tools/dislocation.analyze")
def dislocation(req: DislocationReq):
    try:
        band = timellm.predict_band(req.companyName, req.analysisDate, 20)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="해당 종목 데이터가 없습니다.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # 임시 discrepancy (룰/지표가 붙으면 대체)
    discrepancy = {
        "score": 70.0,
        "trend": "POSITIVE",
        "summary": "임시 요약: 더미 밴드 기준 긍정적 괴리."
    }
    return {
        "companyName": req.companyName,
        "analysisDate": req.analysisDate,
        "discrepancy": discrepancy,
        "timellm_band": band,
        "causalFactors": []
    }
