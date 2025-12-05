from pydantic import BaseModel

class Tool1StockInfoRequest(BaseModel):
    ticker: str
    start_date: str  # "YYYY-MM-DD"
    end_date: str    # "YYYY-MM-DD"
