    from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI


def _safe_json_loads(text: str) -> Optional[dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def _read_json_file(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


CLASS_PERSONALITIES: dict[str, str] = {
    "pengawal": "protective and cautious, values security",
    "pemburu": "patient and opportunistic, values good setups",
    "pendekar": "bold and aggressive, values action",
    "dukun": "analytical and mystical, values patterns",
}


@dataclass
class LLMGenerator:
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")

        base_dir = os.path.dirname(os.path.dirname(__file__))
        self._coaching_templates = _read_json_file(os.path.join(base_dir, "data", "coaching_templates.json"))
        self._quest_templates = _read_json_file(os.path.join(base_dir, "data", "quest_templates.json"))

    def _call_openai_sync(self, prompt: str, max_tokens: int = 220) -> Optional[dict[str, Any]]:
        if not self.api_key:
            return None

        try:
            client = OpenAI(api_key=self.api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Return ONLY valid JSON. No markdown. No extra keys.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.4,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content
            if not text:
                return None
            return _safe_json_loads(text)
        except Exception:
            return None

    async def _call_openai(self, prompt: str, max_tokens: int = 220) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(self._call_openai_sync, prompt, max_tokens)

    def _template_coaching(self, behaviour_state: str, trader_class: str, trader_name: str) -> dict[str, Any]:
        tpl = self._coaching_templates.get(behaviour_state) or self._coaching_templates.get("disciplined")
        class_name = trader_class.capitalize()
        dialogue = tpl["dialogue"].format(name=trader_name, class_name=class_name)
        return {
            "dialogue": dialogue,
            "expression": tpl.get("expression", "neutral"),
            "severity": "high" if tpl.get("advice_type") == "stop_trading" else "medium",
            "advice_type": tpl.get("advice_type", "continue"),
            "xp_change": int(tpl.get("xp_change", 0)),
            "hp_change": int(tpl.get("hp_change", 0)),
            "follow_up_prompt": tpl.get("follow_up_prompt", ""),
        }

    async def generate_coaching(
        self,
        behaviour_state: str,
        behaviour_confidence: float,
        market_analysis: dict[str, Any],
        trader_class: str,
        trader_name: str,
        context: str,
    ) -> dict[str, Any]:
        personality = CLASS_PERSONALITIES.get(trader_class, "")
        market_bias = market_analysis.get("market_bias", market_analysis.get("trend", "unclear"))
        setup_quality = market_analysis.get("setup_quality")
        risk_flags = market_analysis.get("risk_flags")
        rsi_value = market_analysis.get("rsi", market_analysis.get("indicators", {}).get("rsi", 50))
        prompt = f"""You are Tok Bomoh, an ancient village shaman who guides young traders through the spirit world. You speak with wisdom, using tribal metaphors about hunting, spirits, and village life.

Trader: {trader_name} ({trader_class}; {personality}).
Behaviour: {behaviour_state} (confidence {behaviour_confidence:.2f}).
Market: trend {market_analysis.get('trend','unclear')}, bias {market_bias}, setup_quality {setup_quality}, risk_flags {risk_flags}, rsi {rsi_value}.
Context: {context}.

Generate a short coaching message (2-3 sentences). Be warm but firm. Reference Hantu Tamak if risk is high.

Respond ONLY as JSON with keys:
{{
  \"dialogue\": str,
  \"expression\": \"warning\"|\"happy\"|\"concerned\"|\"proud\"|\"neutral\"|\"mystical\",
  \"advice_type\": \"continue\"|\"caution\"|\"stop_trading\",
  \"xp_change\": int,
  \"hp_change\": int,
  \"follow_up_prompt\": str
}}
"""

        result = await self._call_openai(prompt)
        if not result or "dialogue" not in result:
            return self._template_coaching(behaviour_state, trader_class, trader_name)

        advice_type = result.get("advice_type", "continue")
        severity = "high" if advice_type == "stop_trading" else "medium" if advice_type == "caution" else "low"
        return {
            "dialogue": str(result.get("dialogue", "")),
            "expression": str(result.get("expression", "neutral")),
            "severity": severity,
            "advice_type": advice_type,
            "xp_change": int(result.get("xp_change", 0)),
            "hp_change": int(result.get("hp_change", 0)),
            "follow_up_prompt": str(result.get("follow_up_prompt", "")),
        }

    def _template_quest(
        self,
        market_analysis: dict[str, Any],
        trader_level: int,
        trader_class: str,
        quest_type: str,
    ) -> dict[str, Any]:
        patterns = market_analysis.get("patterns")
        if isinstance(patterns, list) and patterns and isinstance(patterns[0], dict):
            pattern_name = patterns[0].get("name")
        elif isinstance(patterns, list) and patterns:
            pattern_name = patterns[0]
        else:
            pattern_name = "consolidation"

        qt = self._quest_templates.get(quest_type, {})
        tpl = qt.get(pattern_name) or qt.get("consolidation")
        if not tpl:
            tpl = {
                "title": "The Trading Grounds",
                "narrative": "The spirits are calm today. Read the signs and choose wisely.",
                "correct": "Wait for a clear break — the village must not waste Semangat",
                "wrongs": ["Strike blindly", "Ignore the signs"],
            }

        choices = [
            {"id": "a", "text": tpl["correct"], "correct": True, "feedback": "Good eyes. The pattern speaks clearly."},
            {"id": "b", "text": tpl["wrongs"][0], "correct": False, "feedback": "Not quite. Look again at the structure."},
            {"id": "c", "text": tpl["wrongs"][1], "correct": False, "feedback": "The spirits reward patience and clarity."},
        ]

        return {
            "quest_id": f"village_{quest_type}_{pattern_name}_{trader_level}",
            "quest_title": tpl["title"],
            "quest_narrative": tpl["narrative"],
            "pattern_shown": pattern_name,
            "chart_description": f"Pattern detected: {pattern_name}",
            "choices": choices,
            "xp_reward": int(30 + 10 * max(trader_level, 1)),
            "correct_answer": "a",
        }

    async def generate_quest(
        self,
        market_analysis: dict[str, Any],
        trader_level: int,
        trader_class: str,
        quest_type: str,
    ) -> dict[str, Any]:
        prompt = f"""You are Tok Bomoh in Kampung Dagangan. Create a short quest for a trading-learning game.

Inputs:
- trader_level: {trader_level}
- trader_class: {trader_class}
- quest_type: {quest_type}
- market_analysis: {json.dumps(market_analysis)[:800]}

Respond ONLY as JSON with keys:
quest_id, quest_title, quest_narrative, pattern_shown, chart_description, choices (list of 3), xp_reward, correct_answer.
Each choice: id (a/b/c), text, correct (bool), feedback.
"""

        result = await self._call_openai(prompt)
        if not result or "choices" not in result:
            return self._template_quest(market_analysis, trader_level, trader_class, quest_type)
        return result

    async def generate_share(
        self,
        achievement_type: str,
        boss_name: Optional[str],
        trader_name: str,
        trader_class: str,
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        boss = boss_name or "Hantu Tamak"

        prompt = f"""Write a social share text for a game achievement.
Theme: Kampung Dagangan, Tok Bomoh, tribal trading.
Achievement type: {achievement_type}
Boss: {boss}
Trader: {trader_name} ({trader_class})
Stats: {json.dumps(stats)}

Respond ONLY as JSON with keys: title, share_text, short_text, stats_display (line1,line2,line3).
"""

        result = await self._call_openai(prompt)
        if result and "share_text" in result:
            return result

        xp = stats.get("xp_earned") or stats.get("xp") or 0
        discipline = stats.get("discipline_score") or stats.get("discipline")
        rounds = stats.get("rounds_won") or stats.get("rounds")

        title = f"Hantu Tamak Defeated!" if achievement_type == "boss_defeated" else "Trade Quest Achievement!"
        share_text = (
            f"{trader_name} the {trader_class.capitalize()} has triumphed in Kampung Dagangan!\n\n"
            f"XP earned: {xp}\n"
            + (f"Discipline: {discipline}%\n" if discipline is not None else "")
            + (f"Rounds won: {rounds}\n" if rounds is not None else "")
            + "\nThe spirits witness this resolve.\n\n#TradeQuest #TradingPsychology"
        )

        stats_display = {
            "line1": f"Combat Rounds: {rounds}" if rounds is not None else "Combat Rounds: -",
            "line2": f"XP Earned: {xp}",
            "line3": f"Discipline: {discipline}%" if discipline is not None else "Discipline: -",
        }

        return {
            "title": title,
            "share_text": share_text,
            "short_text": "I unlocked an achievement in Trade Quest!",
            "stats_display": stats_display,
        }

    async def generate_loss_explanation(
        self,
        trade: dict[str, Any],
        market_analysis: dict[str, Any],
        behaviour_analysis: dict[str, Any],
        trader_class: str,
        trader_name: str,
    ) -> dict[str, Any]:
        """Generate educational explanation for a losing trade."""
        personality = CLASS_PERSONALITIES.get(trader_class, "")
        market_bias = market_analysis.get("market_bias", market_analysis.get("trend", "unclear"))
        trend = market_analysis.get("trend", "neutral")
        patterns = market_analysis.get("patterns", [])
        behaviour_state = behaviour_analysis.get("primary_state", "unknown")
        risk_level = behaviour_analysis.get("risk_level", "low")
        reasoning = behaviour_analysis.get("reasoning", "")

        prompt = f"""You are Tok Bomoh, an ancient village shaman explaining a trading loss to a student.
Speak with wisdom, using tribal metaphors about hunting, spirits, and village life.

Trader: {trader_name} ({trader_class}; {personality}).

Trade Details:
- Action: {trade.get('action')}
- Symbol: {trade.get('symbol')}
- Size: {trade.get('size')}
- PnL: {trade.get('pnl')}

Market Conditions at Trade:
- Trend: {trend}
- Market Bias: {market_bias}
- Patterns: {', '.join(p.get('name') for p in patterns[:3]) if patterns else 'none'}

Behaviour Analysis:
- State: {behaviour_state}
- Risk Level: {risk_level}
- Reasoning: {reasoning}

Generate an educational post-mortem analysis. Be gentle but clear.
Focus on what the trader can LEARN, not blame.

Respond ONLY as JSON with keys:
{{
  "market_conditions": str,
  "behaviour_mistake": str,
  "lesson_learned": str,
  "next_time_suggestion": str,
  "tok_bomoh_wisdom": str
}}"""

        result = await self._call_openai(prompt, max_tokens=400)
        if result and all(k in result for k in ["market_conditions", "lesson_learned"]):
            return result

        # Template fallback
        return self._template_loss_explanation(
            trade, market_analysis, behaviour_analysis, trader_name
        )

    def _template_loss_explanation(
        self,
        trade: dict[str, Any],
        market_analysis: dict[str, Any],
        behaviour_analysis: dict[str, Any],
        trader_name: str,
    ) -> dict[str, Any]:
        """Fallback template for loss explanation when LLM fails."""
        trend = market_analysis.get("trend", "neutral")
        patterns = market_analysis.get("patterns", [])
        behaviour_state = behaviour_analysis.get("primary_state", "unknown")
        risk_level = behaviour_analysis.get("risk_level", "low")

        market_conditions = f"The market showed {trend} conditions."
        if patterns:
            pattern_names = [p.get("name") for p in patterns[:2]]
            market_conditions += f" Patterns present: {', '.join(pattern_names)}."

        behaviour_mistake = {
            "revenge_trading": "Trading larger size after a previous loss to 'make it back'.",
            "fomo": "Entering too quickly, fearing to miss out on a move.",
            "overtrading": "Too many trades in quick succession, fatigue setting in.",
            "tilt": "Emotional trading after a losing streak, decisions clouded.",
            "greed": "Position too large for the setup quality.",
            "fear": "Exiting too early or hesitating on good setups.",
            "disciplined": "Even disciplined hunters miss sometimes. The setup was valid.",
        }.get(behaviour_state, "Standard loss within risk parameters.")

        lesson_learned = {
            "revenge_trading": "Losses are part of the hunt. Wait for the next proper signal.",
            "fomo": "The jungle rewards patience. Wait for prey to come to you.",
            "overtrading": "Rest between hunts. A tired hunter makes mistakes.",
            "tilt": "Step away when Semangat (spirit) is stormy. Return when calm.",
            "greed": "Size must match the prey. Don't hunt rabbits with elephant spears.",
            "fear": "Trust your training. The spirits guide those who trust their path.",
            "disciplined": "Good process, bad outcome. This happens to the best hunters.",
        }.get(behaviour_state, "Every loss carries a lesson. Listen to it.")

        next_time_suggestion = {
            "revenge_trading": "Take 5 breaths after a loss before entering again.",
            "fomo": "Set a timer. Wait 2 minutes before chasing any move.",
            "overtrading": "Maximum 3 trades per session. Quality over quantity.",
            "tilt": "Stop trading for the session. Return tomorrow.",
            "greed": "Reduce size by half. Focus on technique, not catch size.",
            "fear": "Review your plan before entering. Confidence comes from preparation.",
            "disciplined": "Continue your disciplined approach. Results follow process.",
        }.get(behaviour_state, "Reflect on this trade before your next hunt.")

        tok_bomoh_wisdom = f"""{trader_name}, the spirits teach us through both victory and defeat.
This loss is not a failure—it is tuition paid to the market.

The {trend} winds were blowing, yet {'the signs were unclear' if risk_level == 'high' else 'you read them well'}.
Even the best hunters return empty-handed sometimes.

Remember: one bad hunt does not make a bad hunter. But ignoring the lesson does.
"""

        return {
            "market_conditions": market_conditions,
            "behaviour_mistake": behaviour_mistake,
            "lesson_learned": lesson_learned,
            "next_time_suggestion": next_time_suggestion,
            "tok_bomoh_wisdom": tok_bomoh_wisdom,
        }

