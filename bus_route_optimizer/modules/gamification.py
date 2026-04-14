"""
Gamification Module: Scoring, achievements, and leaderboard for Bus Route Optimizer.

Turns every optimisation run into a scored "game session" with:
- Efficiency Score  (0-100)
- Star Rating       (1-5 ⭐)
- Badges / Achievements
- Session Leaderboard (persisted to JSON)
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

# ─────────────────────────── Achievement definitions ────────────────────────

ACHIEVEMENTS = {
    "first_run":        {"name": "🚌 First Run",      "desc": "Complete your first optimisation",          "xp": 50},
    "speed_demon":      {"name": "⚡ Speed Demon",    "desc": "Finish optimisation in under 2 seconds",    "xp": 75},
    "fuel_saver":       {"name": "🌿 Fuel Saver",     "desc": "Keep total fuel under 45 litres",           "xp": 100},
    "efficiency_ace":   {"name": "🎯 Efficiency Ace", "desc": "Score 80 or above",                         "xp": 150},
    "route_master":     {"name": "🏆 Route Master",   "desc": "Score 90 or above",                         "xp": 250},
    "perfect_score":    {"name": "💎 Perfect Score",  "desc": "Score 95 or above",                         "xp": 500},
    "cluster_king":     {"name": "👑 Cluster King",   "desc": "Run with 5 or more buses",                  "xp": 120},
    "minimalist":       {"name": "🎀 Minimalist",     "desc": "Optimise 10 stops with 1 bus",              "xp": 80},
    "dp_devotee":       {"name": "🧮 DP Devotee",     "desc": "Use dynamic-programming routing",           "xp": 90},
    "hybrid_hero":      {"name": "🦸 Hybrid Hero",    "desc": "Use hybrid routing",                        "xp": 70},
    "run_5":            {"name": "🔁 Consistent",     "desc": "Complete 5 optimisation sessions",          "xp": 100},
    "run_10":           {"name": "🔥 On Fire",        "desc": "Complete 10 optimisation sessions",         "xp": 200},
}


class GamificationEngine:
    """Compute scores, award badges, and maintain a leaderboard."""

    LEADERBOARD_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "leaderboard.json")

    def __init__(self):
        os.makedirs(os.path.dirname(self.LEADERBOARD_FILE), exist_ok=True)
        self.leaderboard: List[Dict] = self._load_leaderboard()

    # ──────────────────────────── Scoring ────────────────────────────────────

    def compute_score(
        self,
        total_distance_km: float,
        total_time_min: float,
        total_fuel_litres: float,
        num_buses: int,
        num_stops: int,
        elapsed_seconds: float,
    ) -> Dict:
        """
        Compute a composite efficiency score (0-100) and star rating.

        Components:
          • distance_score  – fewer km per stop is better
          • time_score      – fewer minutes per stop is better
          • fuel_score      – lower fuel is better
          • speed_bonus     – bonus if elapsed wall-clock time < 3 s
        """
        stops_per_bus = max(num_stops / max(num_buses, 1), 1)

        # Distance: target ≤ 10 units/stop → 100 pts; ≥ 30 units/stop → 0 pts
        # (on a default 100×100 map, typical inter-stop spacing ≈ 10-18 units)
        km_per_stop = total_distance_km / max(num_stops, 1)
        dist_score = max(0.0, 100.0 - (km_per_stop - 10.0) * (100.0 / 20.0))

        # Time: target ≤ 20 min/stop → 100 pts; ≥ 70 min/stop → 0 pts
        # (using estimate_travel_time speed of 30 units/h: 10 units ≈ 20 min)
        min_per_stop = total_time_min / max(num_stops, 1)
        time_score = max(0.0, 100.0 - (min_per_stop - 20.0) * (100.0 / 50.0))

        # Fuel: target ≤ 50 L → 100 pts; ≥ 150 L → 0 pts
        # (using estimate_fuel_consumption at 5 units/L: 250 units → 50 L)
        fuel_score = max(0.0, 100.0 - (total_fuel_litres - 50.0) * (100.0 / 100.0))

        # Wall-clock speed bonus (max 10 pts)
        speed_bonus = max(0.0, 10.0 - elapsed_seconds * 2.0)

        raw = (dist_score * 0.40 + time_score * 0.30 + fuel_score * 0.20 + speed_bonus * 0.10)
        score = min(100.0, max(0.0, round(raw, 1)))

        stars = (
            5 if score >= 90 else
            4 if score >= 75 else
            3 if score >= 55 else
            2 if score >= 35 else
            1
        )

        return {
            "score": score,
            "stars": stars,
            "star_display": "⭐" * stars + "☆" * (5 - stars),
            "dist_score": round(dist_score, 1),
            "time_score": round(time_score, 1),
            "fuel_score": round(fuel_score, 1),
            "speed_bonus": round(speed_bonus, 1),
        }

    # ──────────────────────────── Achievements ───────────────────────────────

    def evaluate_achievements(
        self,
        score: float,
        elapsed_seconds: float,
        total_fuel: float,
        num_buses: int,
        num_stops: int,
        routing_method: str,
        session_count: int,
    ) -> List[Dict]:
        """Return list of newly unlocked achievement dicts."""
        earned = []

        def _check(condition: bool, ach_id: str):
            if condition:
                earned.append({**ACHIEVEMENTS[ach_id], "id": ach_id})

        _check(session_count == 1,                 "first_run")
        _check(elapsed_seconds < 2.0,              "speed_demon")
        _check(total_fuel < 45.0,                  "fuel_saver")
        _check(score >= 80,                        "efficiency_ace")
        _check(score >= 90,                        "route_master")
        _check(score >= 95,                        "perfect_score")
        _check(num_buses >= 5,                     "cluster_king")
        _check(num_stops <= 10 and num_buses == 1, "minimalist")
        _check(routing_method == "dp",             "dp_devotee")
        _check(routing_method == "hybrid",         "hybrid_hero")
        _check(session_count == 5,                 "run_5")
        _check(session_count == 10,                "run_10")

        return earned

    # ──────────────────────────── Gallery helper ──────────────────────────────

    @staticmethod
    def all_achievements() -> List[Dict]:
        """Return every achievement definition with its id attached."""
        return [{**v, "id": k} for k, v in ACHIEVEMENTS.items()]

    # ──────────────────────────── Leaderboard ────────────────────────────────

    def add_to_leaderboard(
        self,
        player_name: str,
        score_data: Dict,
        config: Dict,
        achievements: List[Dict],
    ) -> None:
        """Append a session record and persist to disk."""
        total_xp = sum(a["xp"] for a in achievements)
        entry = {
            "player": player_name,
            "score": score_data["score"],
            "stars": score_data["stars"],
            "xp_earned": total_xp,
            "achievements": [a["name"] for a in achievements],
            "config": config,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        self.leaderboard.append(entry)
        # Keep top-50 by score
        self.leaderboard.sort(key=lambda x: x["score"], reverse=True)
        self.leaderboard = self.leaderboard[:50]
        self._save_leaderboard()

    def get_leaderboard(self, top_n: int = 10) -> List[Dict]:
        return self.leaderboard[:top_n]

    def get_session_count(self) -> int:
        return len(self.leaderboard)

    # ──────────────────────────── Persistence ────────────────────────────────

    def _load_leaderboard(self) -> List[Dict]:
        if os.path.exists(self.LEADERBOARD_FILE):
            try:
                with open(self.LEADERBOARD_FILE) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def _save_leaderboard(self) -> None:
        with open(self.LEADERBOARD_FILE, "w") as f:
            json.dump(self.leaderboard, f, indent=2)
