from fastapi import FastAPI
from app.routers import orchestrator, text_analysis, voice_analysis, fusion

app = FastAPI(title="Depression Detection ML Engine")

app.include_router(orchestrator.router, prefix="/orchestrator", tags=["orchestrator"])
app.include_router(text_analysis.router, prefix="/analyze/text", tags=["text"])
app.include_router(voice_analysis.router, prefix="/analyze/voice", tags=["voice"])
app.include_router(fusion.router, prefix="/fusion", tags=["fusion"])


@app.get("/health")
def health():
    return {"status": "ok"}
