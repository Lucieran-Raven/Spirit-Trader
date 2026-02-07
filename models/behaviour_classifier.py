from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def extract_features(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    if not decisions:
        return {
            "avg_size_change_after_loss": 1.0,
            "avg_size_change_after_win": 1.0,
            "avg_interval": 0.0,
            "interval_decreasing": False,
            "win_rate": 0.5,
            "recent_win_rate": 0.5,
            "loss_chasing": False,
            "max_drawdown": 0.0,
            "in_drawdown": False,
            "trade_count": 0,
            "direction_flips": 0,
        }

    sizes = [float(d.get("size", 0.0)) for d in decisions]
    results = [d.get("result") for d in decisions]
    timestamps = [int(d.get("timestamp", 0)) for d in decisions]
    actions = [str(d.get("action", "")).lower() for d in decisions]

    sizes_after_loss: list[float] = []
    sizes_after_win: list[float] = []

    for i in range(1, len(decisions)):
        prev_size = sizes[i - 1] if sizes[i - 1] != 0 else 1.0
        if results[i - 1] == "loss":
            sizes_after_loss.append(sizes[i] / prev_size)
        elif results[i - 1] == "win":
            sizes_after_win.append(sizes[i] / prev_size)

    intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

    wins = sum(1 for r in results if r == "win")
    total = sum(1 for r in results if r is not None)

    recent = decisions[-5:]
    recent_wins = sum(1 for d in recent if d.get("result") == "win")
    recent_total = sum(1 for d in recent if d.get("result") is not None)

    loss_chasing = False
    for i in range(1, len(decisions)):
        if (
            results[i - 1] == "loss"
            and actions[i] == actions[i - 1]
            and (timestamps[i] - timestamps[i - 1]) < 300
        ):
            loss_chasing = True
            break

    cumulative_pnl: list[float] = []
    running = 0.0
    for d in decisions:
        pnl = d.get("pnl")
        if pnl is not None:
            running += float(pnl)
        cumulative_pnl.append(running)

    direction_flips = 0
    for i in range(1, len(actions)):
        if actions[i] and actions[i - 1] and actions[i] != actions[i - 1]:
            direction_flips += 1

    return {
        "avg_size_change_after_loss": float(np.mean(sizes_after_loss)) if sizes_after_loss else 1.0,
        "avg_size_change_after_win": float(np.mean(sizes_after_win)) if sizes_after_win else 1.0,
        "avg_interval": float(np.mean(intervals)) if intervals else 0.0,
        "interval_decreasing": bool(intervals[-1] < intervals[0]) if len(intervals) > 1 else False,
        "win_rate": float(wins / total) if total > 0 else 0.5,
        "recent_win_rate": float(recent_wins / recent_total) if recent_total > 0 else 0.5,
        "loss_chasing": bool(loss_chasing),
        "max_drawdown": float(min(cumulative_pnl)) if cumulative_pnl else 0.0,
        "in_drawdown": bool(cumulative_pnl[-1] < 0) if cumulative_pnl else False,
        "trade_count": int(len(decisions)),
        "direction_flips": int(direction_flips),
    }


def _risk_level(state: str) -> str:
    if state in {"tilt"}:
        return "critical"
    if state in {"revenge_trading", "fomo", "greed", "overconfident"}:
        return "high"
    if state in {"overtrading", "fear"}:
        return "medium"
    return "low"


def classify_state(features: dict[str, Any]) -> tuple[str, float, list[tuple[str, float]]]:
    avg_loss = float(features.get("avg_size_change_after_loss", 1.0))
    avg_win = float(features.get("avg_size_change_after_win", 1.0))
    avg_interval = float(features.get("avg_interval", 0.0))
    interval_decreasing = bool(features.get("interval_decreasing", False))
    in_drawdown = bool(features.get("in_drawdown", False))
    loss_chasing = bool(features.get("loss_chasing", False))
    recent_wr = float(features.get("recent_win_rate", 0.5))
    wr = float(features.get("win_rate", 0.5))
    direction_flips = int(features.get("direction_flips", 0))
    trade_count = int(features.get("trade_count", 0))

    secondary: list[tuple[str, float]] = []

    tilt_score = 0.0
    tilt_score += 0.4 if direction_flips >= 3 else 0.0
    tilt_score += 0.3 if avg_interval and avg_interval < 45 else 0.0
    tilt_score += 0.3 if avg_loss >= 2.0 and in_drawdown else 0.0

    if tilt_score >= 0.75:
        secondary.extend([("revenge_trading", 0.65), ("overtrading", 0.6)])
        return "tilt", min(0.95, tilt_score), secondary

    if avg_loss > 1.5 and in_drawdown and (interval_decreasing or loss_chasing):
        if avg_interval < 120:
            secondary.append(("overtrading", 0.65))
        if avg_loss > 2.2:
            secondary.append(("poor_sizing", 0.72))
        return "revenge_trading", 0.87, secondary

    if interval_decreasing and avg_interval < 60 and recent_wr < 0.35 and trade_count >= 3:
        secondary.append(("overtrading", 0.6))
        return "fomo", 0.78, secondary

    if avg_interval and avg_interval < 120 and trade_count >= 5:
        return "overtrading", 0.72, secondary

    if recent_wr > wr and avg_win < 0.85 and trade_count >= 3:
        return "fear", 0.66, secondary

    if avg_win > 1.7 and recent_wr > 0.55:
        secondary.append(("greed", 0.62))
        return "overconfident", 0.73, secondary

    return "disciplined", 0.62, secondary


@dataclass
class BehaviourClassifier:
    def classify(self, decisions: list[dict[str, Any]], trader_class: str, current_pnl: float) -> dict[str, Any]:
        features = extract_features(decisions)
        state, confidence, secondary = classify_state(features)

        risk_level = _risk_level(state)
        recommendation = "continue"
        if risk_level in {"high", "critical"}:
            recommendation = "stop_trading"
        elif risk_level == "medium":
            recommendation = "caution"

        reasoning_parts: list[str] = []
        if features.get("avg_size_change_after_loss", 1.0) > 1.5:
            reasoning_parts.append("Position size increased after losses")
        if features.get("interval_decreasing"):
            reasoning_parts.append("Time between trades is decreasing")
        if features.get("loss_chasing"):
            reasoning_parts.append("Re-entered quickly after a loss")
        if features.get("recent_win_rate", 0.5) < 0.3:
            reasoning_parts.append("Recent win rate is low")
        if not reasoning_parts:
            reasoning_parts.append("No major risk flags detected")

        secondary_states = [{"state": s, "confidence": float(round(c, 2))} for s, c in secondary]

        api_features = {
            "position_size_delta": float(round(features.get("avg_size_change_after_loss", 1.0) - 1.0, 2)),
            "trade_frequency_increase": float(
                round((120.0 / max(features.get("avg_interval", 120.0), 1.0)), 2)
            ),
            "loss_chasing": bool(features.get("loss_chasing", False)),
            "stop_loss_moved": False,
            "win_rate_recent": float(round(features.get("recent_win_rate", 0.5), 2)),
        }

        return {
            "primary_state": state,
            "confidence": float(round(confidence, 2)),
            "secondary_states": secondary_states,
            "risk_level": risk_level,
            "reasoning": ". ".join(reasoning_parts) + ".",
            "features_detected": api_features,
            "recommendation": recommendation,
        }
