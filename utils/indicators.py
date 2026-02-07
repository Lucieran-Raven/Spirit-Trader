from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def calculate_moving_average(closes: list[float], period: int) -> float:
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(closes) < period:
        return float(np.mean(closes)) if closes else 0.0
    return float(np.mean(closes[-period:]))


def calculate_rsi(closes: list[float], period: int = 14) -> float:
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(np.array(closes, dtype=float))
    gains = np.clip(deltas, 0, None)
    losses = -np.clip(deltas, None, 0)

    window_gains = gains[-period:]
    window_losses = losses[-period:]

    avg_gain = float(np.mean(window_gains))
    avg_loss = float(np.mean(window_losses))

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(np.clip(rsi, 0.0, 100.0))


def calculate_volatility(closes: list[float], period: int = 20) -> float:
    if len(closes) < 2:
        return 0.0
    arr = np.array(closes, dtype=float)
    returns = np.diff(np.log(arr + 1e-12))
    if len(returns) == 0:
        return 0.0
    if len(returns) >= period:
        returns = returns[-period:]
    return float(np.std(returns))


@dataclass(frozen=True)
class LevelCluster:
    level: float
    hits: int


def _swing_points(values: list[float], lookback: int = 2) -> list[float]:
    if len(values) < (lookback * 2 + 1):
        return []
    swings: list[float] = []
    for i in range(lookback, len(values) - lookback):
        left = values[i - lookback : i]
        right = values[i + 1 : i + 1 + lookback]
        v = values[i]
        if all(v > x for x in left) and all(v > x for x in right):
            swings.append(v)
        elif all(v < x for x in left) and all(v < x for x in right):
            swings.append(v)
    return swings


def _cluster_levels(levels: Iterable[float], sensitivity: float) -> list[LevelCluster]:
    levels_sorted = sorted(float(x) for x in levels)
    clusters: list[LevelCluster] = []

    for lvl in levels_sorted:
        if not clusters:
            clusters.append(LevelCluster(level=lvl, hits=1))
            continue

        last = clusters[-1]
        if last.level == 0:
            rel = abs(lvl - last.level)
        else:
            rel = abs(lvl - last.level) / abs(last.level)

        if rel <= sensitivity:
            new_level = (last.level * last.hits + lvl) / (last.hits + 1)
            clusters[-1] = LevelCluster(level=float(new_level), hits=last.hits + 1)
        else:
            clusters.append(LevelCluster(level=lvl, hits=1))

    clusters.sort(key=lambda c: (-c.hits, c.level))
    return clusters


def find_support_resistance(candles: list[dict], sensitivity: float = 0.02) -> dict:
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]

    swing_highs = [x for x in _swing_points(highs, lookback=2)]
    swing_lows = [x for x in _swing_points(lows, lookback=2)]

    clustered_highs = _cluster_levels(swing_highs, sensitivity=sensitivity)
    clustered_lows = _cluster_levels(swing_lows, sensitivity=sensitivity)

    resistance = [c.level for c in clustered_highs[:3]]
    support = [c.level for c in clustered_lows[:3]]

    resistance.sort()
    support.sort(reverse=False)

    return {
        "support_levels": [float(x) for x in support],
        "resistance_levels": [float(x) for x in resistance],
    }


def detect_trend(closes: list[float]) -> tuple[str, float]:
    if len(closes) < 10:
        return "neutral", 0.0

    ma_short = calculate_moving_average(closes, period=min(20, len(closes)))
    ma_long = calculate_moving_average(closes, period=min(50, len(closes)))

    diff = ma_short - ma_long
    base = abs(ma_long) if ma_long != 0 else 1.0
    ma_strength = min(abs(diff) / base * 5.0, 1.0)

    recent = closes[-10:]
    slope = (recent[-1] - recent[0]) / (abs(recent[0]) + 1e-12)
    slope_strength = min(abs(slope) * 10.0, 1.0)

    strength = float(np.clip(0.6 * ma_strength + 0.4 * slope_strength, 0.0, 1.0))

    if diff > 0 and slope > 0:
        return "bullish", strength
    if diff < 0 and slope < 0:
        return "bearish", strength
    return "neutral", strength * 0.5
