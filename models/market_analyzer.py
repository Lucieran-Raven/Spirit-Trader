from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.indicators import (
    calculate_moving_average,
    calculate_rsi,
    calculate_volatility,
    detect_trend,
    find_support_resistance,
)


def _rsi_signal(rsi: float) -> str:
    if rsi >= 70:
        return "overbought"
    if rsi <= 30:
        return "oversold"
    return "neutral"


def _ma_cross(ma20: float, ma50: float) -> str:
    if ma20 > ma50:
        return "bullish"
    if ma20 < ma50:
        return "bearish"
    return "neutral"


def _volume_trend(candles: list[dict]) -> str:
    volumes = [c.get("volume") for c in candles]
    vols = [float(v) for v in volumes if v is not None]
    if len(vols) < 6:
        return "unknown"
    first = float(np.mean(vols[: len(vols) // 2]))
    second = float(np.mean(vols[len(vols) // 2 :]))
    if second > first * 1.05:
        return "increasing"
    if second < first * 0.95:
        return "decreasing"
    return "flat"


def detect_patterns(candles: list[dict]) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    if len(candles) < 15:
        return patterns

    recent = candles[-10:]
    highs = [float(c["high"]) for c in recent]
    lows = [float(c["low"]) for c in recent]
    closes = [float(c["close"]) for c in recent]

    if all(highs[i] < highs[i + 1] for i in range(len(highs) - 1)):
        patterns.append(
            {
                "name": "higher_highs",
                "confidence": 0.85,
                "description": "Price making consecutive higher highs",
            }
        )

    if all(lows[i] > lows[i + 1] for i in range(len(lows) - 1)):
        patterns.append(
            {
                "name": "lower_lows",
                "confidence": 0.85,
                "description": "Price making consecutive lower lows",
            }
        )

    sr = find_support_resistance(candles, sensitivity=0.02)
    support_levels = sr.get("support_levels", [])
    resistance_levels = sr.get("resistance_levels", [])
    last_close = float(candles[-1]["close"])
    prev_close = float(candles[-2]["close"]) if len(candles) >= 2 else last_close

    if support_levels:
        nearest_support = min(support_levels, key=lambda x: abs(last_close - x))
        if prev_close < nearest_support * 1.01 and last_close > prev_close:
            patterns.append(
                {
                    "name": "support_bounce",
                    "confidence": 0.68,
                    "description": f"Price bounced off support level at {nearest_support:.4f}",
                }
            )

    if resistance_levels:
        nearest_res = min(resistance_levels, key=lambda x: abs(last_close - x))
        if prev_close > nearest_res * 0.99 and last_close < prev_close:
            patterns.append(
                {
                    "name": "resistance_rejection",
                    "confidence": 0.66,
                    "description": f"Price rejected near resistance at {nearest_res:.4f}",
                }
            )

    range_pct = (max(highs) - min(lows)) / (abs(np.mean(closes)) + 1e-12)
    if range_pct < 0.01:
        patterns.append(
            {
                "name": "consolidation",
                "confidence": 0.7,
                "description": "Price is ranging in a tight consolidation",
            }
        )

    return patterns


@dataclass
class MarketAnalyzer:
    def analyze(self, candles: list[dict], timeframe: str = "1h") -> dict[str, Any]:
        if not candles or len(candles) < 5:
            raise ValueError("Need at least 5 candles")

        closes = [float(c["close"]) for c in candles]
        rsi = calculate_rsi(closes, period=14)
        ma20 = calculate_moving_average(closes, period=20)
        ma50 = calculate_moving_average(closes, period=50)
        trend, strength = detect_trend(closes)
        volatility = calculate_volatility(closes, period=20)

        sr = find_support_resistance(candles, sensitivity=0.02)
        patterns = detect_patterns(candles)

        ma_cross = _ma_cross(ma20, ma50)
        rsi_sig = _rsi_signal(rsi)

        last_close = closes[-1]

        score = 0.0
        score += 0.6 * (strength if trend == "bullish" else -strength if trend == "bearish" else 0.0)
        score += 0.25 * (1.0 if ma_cross == "bullish" else -1.0 if ma_cross == "bearish" else 0.0)
        if rsi_sig == "oversold":
            score += 0.15
        elif rsi_sig == "overbought":
            score -= 0.15

        for p in patterns:
            if p["name"] in {"higher_highs", "support_bounce", "breakout"}:
                score += 0.05 * float(p.get("confidence", 0.6))
            if p["name"] in {"lower_lows", "resistance_rejection", "breakdown"}:
                score -= 0.05 * float(p.get("confidence", 0.6))

        market_bias = "neutral"
        if score > 0.15:
            market_bias = "bullish"
        elif score < -0.15:
            market_bias = "bearish"

        setup_quality = float(np.clip(abs(score), 0.0, 1.0))

        risk_flags: list[str] = []
        if volatility > 0.03:
            risk_flags.append("high_volatility")
        if rsi_sig in {"overbought", "oversold"}:
            risk_flags.append(f"rsi_{rsi_sig}")
        if trend == "neutral" or strength < 0.25:
            risk_flags.append("range_market")

        reasoning_parts: list[str] = []
        reasoning_parts.append(f"{trend.capitalize()} trend")
        reasoning_parts.append(f"RSI {rsi_sig}")
        reasoning_parts.append(f"MA cross {ma_cross}")
        if patterns:
            reasoning_parts.append("patterns: " + ", ".join(p["name"] for p in patterns[:3]))

        indicators = {
            "rsi": float(round(rsi, 2)),
            "rsi_signal": rsi_sig,
            "ma_20": float(round(ma20, 4)),
            "ma_50": float(round(ma50, 4)),
            "ma_cross": ma_cross,
            "support_levels": [float(round(x, 4)) for x in sr.get("support_levels", [])],
            "resistance_levels": [float(round(x, 4)) for x in sr.get("resistance_levels", [])],
            "volatility": float(round(volatility, 6)),
            "volume_trend": _volume_trend(candles),
            "last_close": float(round(last_close, 4)),
            "timeframe": timeframe,
        }

        return {
            "trend": trend,
            "trend_strength": float(round(strength, 2)),
            "patterns": patterns,
            "indicators": indicators,
            "market_bias": market_bias,
            "setup_quality": float(round(setup_quality, 2)),
            "risk_flags": risk_flags,
            "reasoning": "; ".join(reasoning_parts),
        }
