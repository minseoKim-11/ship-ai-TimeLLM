from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.tool1_routes import router as tool1_router
from .api.tool2_routes import router as tool2_router
from .api.tool3_routes import router as tool3_router
from .api.tool4_routes import router as tool4_router



app = FastAPI(
    title="ShipAI Time-LLM API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tool1_router)
app.include_router(tool2_router)
app.include_router(tool3_router)
app.include_router(tool4_router)

@app.get("/")
def root():
    return {"status": "running"}
