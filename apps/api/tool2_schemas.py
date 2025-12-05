from typing import Optional
from pydantic import BaseModel

class Tool2MacroPulseRequest(BaseModel):
    dummy: Optional[str] = None 
