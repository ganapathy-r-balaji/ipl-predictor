"""
IPL Prediction System - Feature Agent
======================================
Responsible for:
  - Engineering match-level features from raw historical data
  - Computing rolling team stats (win rates, form, H2H)
  - Venue-specific performance features
  - Preparing feature matrices for model training & inference
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

DATA_DIR = Path(__file__).parent.parent / "data"


class FeatureAgent:
    """Builds feature matrices for the prediction model."""

    FEATURE_COLS = [
        "t1_overall_wr", "t2_overall_wr",
        "t1_matches_played", "t2_matches_played",
        "t1_recent_wr", "t2_recent_wr",
        "t1_form5", "t2_form5",
        "h2h_t1_wr",
        "t1_venue_wr", "t2_venue_wr",
        "toss_winner_is_t1", "toss_field",
    ]

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None

    def load(self, df: Optional[pd.DataFrame] = None) -> "FeatureAgent":
        """Load historical match data. Pass a DataFrame or load from disk."""
        if df is not None:
            self._df = df
        else:
            path = DATA_DIR / "matches_clean.csv"
            self._df = pd.read_csv(path, parse_dates=["date"])
        self._df = self._df[self._df["no_result"] == 0].copy()
        # Ensure numeric columns
        self._df["winner_runs"] = pd.to_numeric(self._df.get("winner_runs"), errors="coerce")
        self._df["winner_wickets"] = pd.to_numeric(self._df.get("winner_wickets"), errors="coerce")
        return self

    # ── Public API ────────────────────────────────────────────────────────
    def run(self, task: str, **kwargs) -> dict:
        """
        Tasks:
          - 'build_matrix'    → create feature matrix for all matches
          - 'predict_features'→ compute features for a single upcoming match
        """
        if self._df is None:
            self.load()
        if task == "build_matrix":
            return self._build_matrix()
        elif task == "predict_features":
            return self._predict_features(**kwargs)
        return {"error": f"Unknown task: {task}"}

    def build_feature_matrix(self) -> pd.DataFrame:
        """Build and return full feature matrix (used by training pipeline)."""
        result = self._build_matrix()
        return result["dataframe"]

    def get_match_features(self, team1: str, team2: str, venue: str = "",
                           toss_winner: str = "", toss_decision: str = "field"
                           ) -> pd.DataFrame:
        """Return a 1-row feature DataFrame for an upcoming match."""
        feat = self._compute_features(
            team1, team2, venue, toss_winner, toss_decision,
            cutoff_date=None
        )
        return pd.DataFrame([feat])[self.FEATURE_COLS]

    # ── Feature Computation ───────────────────────────────────────────────
    def _build_matrix(self) -> dict:
        rows = []
        df = self._df
        for idx, row in df.iterrows():
            past = df[df["date"] < row["date"]]
            feat = self._compute_features(
                row["team1"], row["team2"], row.get("venue", ""),
                row.get("toss_winner", ""), row.get("toss_decision", ""),
                cutoff_df=past,
            )
            feat["match_id"] = row["match_id"]
            feat["date"] = row["date"]
            feat["season"] = row.get("season_clean", row.get("season", ""))
            feat["team1_won"] = row["team1_won"]
            rows.append(feat)

            if idx % 200 == 0:
                print(f"[FeatureAgent] Processed {idx}/{len(df)} matches")

        out = pd.DataFrame(rows)
        out_path = DATA_DIR / "match_features.csv"
        out.to_csv(out_path, index=False)
        print(f"[FeatureAgent] Feature matrix saved → {out_path}")
        return {"dataframe": out, "rows": len(out), "path": str(out_path)}

    def _predict_features(self, team1: str, team2: str, venue: str = "",
                           toss_winner: str = "", toss_decision: str = "field"
                           ) -> dict:
        feat = self._compute_features(team1, team2, venue, toss_winner, toss_decision)
        return {"features": feat}

    def _compute_features(
        self,
        t1: str, t2: str,
        venue: str, toss_winner: str, toss_decision: str,
        cutoff_date=None, cutoff_df=None,
    ) -> dict:
        past = cutoff_df if cutoff_df is not None else (
            self._df[self._df["date"] < pd.Timestamp(cutoff_date)]
            if cutoff_date else self._df
        )

        wr1, mp1 = self._team_stats(past, t1)
        wr2, mp2 = self._team_stats(past, t2)
        rwr1, _  = self._team_stats(past, t1, lookback=10)
        rwr2, _  = self._team_stats(past, t2, lookback=10)
        f5_1, _  = self._team_stats(past, t1, lookback=5)
        f5_2, _  = self._team_stats(past, t2, lookback=5)
        h2h      = self._h2h_wr(past, t1, t2)
        vwr1     = self._venue_wr(past, t1, venue)
        vwr2     = self._venue_wr(past, t2, venue)

        return {
            "t1_overall_wr":      wr1,
            "t2_overall_wr":      wr2,
            "t1_matches_played":  mp1,
            "t2_matches_played":  mp2,
            "t1_recent_wr":       rwr1,
            "t2_recent_wr":       rwr2,
            "t1_form5":           f5_1,
            "t2_form5":           f5_2,
            "h2h_t1_wr":          h2h,
            "t1_venue_wr":        vwr1,
            "t2_venue_wr":        vwr2,
            "toss_winner_is_t1":  int(toss_winner == t1),
            "toss_field":         int("field" in str(toss_decision).lower()),
        }

    # ── Stat Helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _team_stats(df: pd.DataFrame, team: str, lookback: Optional[int] = None
                    ) -> Tuple[float, int]:
        tm = df[(df["team1"] == team) | (df["team2"] == team)]
        if lookback:
            tm = tm.tail(lookback)
        if len(tm) == 0:
            return 0.5, 0
        wins = (
            ((tm["team1"] == team) & (tm["team1_won"] == 1)).sum()
            + ((tm["team2"] == team) & (tm["team2_won"] == 1)).sum()
        )
        return wins / len(tm), len(tm)

    @staticmethod
    def _h2h_wr(df: pd.DataFrame, t1: str, t2: str) -> float:
        h2h = df[
            ((df["team1"] == t1) & (df["team2"] == t2))
            | ((df["team1"] == t2) & (df["team2"] == t1))
        ]
        if len(h2h) == 0:
            return 0.5
        wins = (
            ((h2h["team1"] == t1) & (h2h["team1_won"] == 1)).sum()
            + ((h2h["team2"] == t1) & (h2h["team2_won"] == 1)).sum()
        )
        return wins / len(h2h)

    @staticmethod
    def _venue_wr(df: pd.DataFrame, team: str, venue: str) -> float:
        if not venue:
            return 0.5
        key = venue[:15]
        vm = df[
            ((df["team1"] == team) | (df["team2"] == team))
            & df["venue"].str.contains(key, na=False, case=False)
        ]
        if len(vm) == 0:
            return 0.5
        wins = (
            ((vm["team1"] == team) & (vm["team1_won"] == 1)).sum()
            + ((vm["team2"] == team) & (vm["team2_won"] == 1)).sum()
        )
        return wins / len(vm)


if __name__ == "__main__":
    agent = FeatureAgent().load()
    print("Testing feature computation...")
    feat = agent.get_match_features(
        "Mumbai Indians", "Chennai Super Kings",
        venue="Wankhede Stadium", toss_winner="Mumbai Indians", toss_decision="bat"
    )
    print(feat.to_string())
