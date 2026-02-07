from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.behaviour_classifier import BehaviourClassifier
from models.llm_generator import LLMGenerator
from models.market_analyzer import MarketAnalyzer

load_dotenv()

app = FastAPI(title="Spirit Trader API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

market_analyzer = MarketAnalyzer()
behaviour_classifier = BehaviourClassifier()
llm_generator = LLMGenerator(api_key=os.getenv("OPENAI_API_KEY"))


class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    timestamp: int
    volume: Optional[float] = None


class MarketAnalysisRequest(BaseModel):
    symbol: str
    candles: List[Candle]
    timeframe: str = "1h"


class Decision(BaseModel):
    action: str
    size: float
    timestamp: int
    price: Optional[float] = None
    result: Optional[str] = None
    pnl: Optional[float] = None


class BehaviourRequest(BaseModel):
    decisions: List[Decision]
    trader_class: str = "pemburu"
    current_pnl: float = 0.0


class CoachingRequest(BaseModel):
    behaviour_state: str
    behaviour_confidence: float
    market_analysis: Dict[str, Any]
    trader_class: str
    trader_name: str
    context: str = "general"


class QuestRequest(BaseModel):
    market_analysis: Dict[str, Any]
    trader_level: int
    trader_class: str
    quest_type: str = "pattern_recognition"


class ShareRequest(BaseModel):
    achievement_type: str
    boss_name: Optional[str] = None
    trader_name: str
    trader_class: str
    stats: Dict[str, Any]


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "alive", "service": "Spirit Trader API"}


@app.post("/analyse-market")
def analyse_market(request: MarketAnalysisRequest) -> dict[str, Any]:
    try:
        candles = [c.model_dump() for c in request.candles]
        result = market_analyzer.analyze(candles, request.timeframe)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify-behaviour")
def classify_behaviour(request: BehaviourRequest) -> dict[str, Any]:
    try:
        decisions = [d.model_dump() for d in request.decisions]
        result = behaviour_classifier.classify(decisions, request.trader_class, request.current_pnl)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-coaching")
async def generate_coaching(request: CoachingRequest) -> dict[str, Any]:
    try:
        result = await llm_generator.generate_coaching(
            request.behaviour_state,
            request.behaviour_confidence,
            request.market_analysis,
            request.trader_class,
            request.trader_name,
            request.context,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-quest")
async def generate_quest(request: QuestRequest) -> dict[str, Any]:
    try:
        result = await llm_generator.generate_quest(
            request.market_analysis,
            request.trader_level,
            request.trader_class,
            request.quest_type,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-share")
async def generate_share(request: ShareRequest) -> dict[str, Any]:
    try:
        result = await llm_generator.generate_share(
            request.achievement_type,
            request.boss_name,
            request.trader_name,
            request.trader_class,
            request.stats,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
