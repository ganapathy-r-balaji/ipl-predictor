"""
IPL Prediction System - Prediction Agent
==========================================
Responsible for:
  - Training the stacked ensemble model (XGBoost + LightGBM + RandomForest)
  - Predicting match outcomes with win probabilities
  - Running Monte Carlo tournament simulation (50,000 runs)
  - Returning per-team championship probabilities
"""

import json, pickle, random, warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"

IPL_2026_TEAMS = [
    "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans",
    "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
    "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

FEATURE_COLS = [
    "t1_overall_wr", "t2_overall_wr",
    "t1_matches_played", "t2_matches_played",
    "t1_recent_wr", "t2_recent_wr",
    "t1_form5", "t2_form5",
    "h2h_t1_wr",
    "t1_venue_wr", "t2_venue_wr",
    "toss_winner_is_t1", "toss_field",
]


class PredictionAgent:
    """Trains ensemble model and generates match + tournament predictions."""

    def __init__(self):
        self.models: Optional[Dict] = None
        self._win_prob_cache: Dict[Tuple, float] = {}
        self._feature_agent = None

    # ── Public API ────────────────────────────────────────────────────────
    def run(self, task: str, **kwargs) -> dict:
        """
        Tasks:
          - 'train'                  → train ensemble on feature matrix
          - 'predict_match'          → predict a single match
          - 'predict_all_fixtures'   → predict all IPL 2026 fixtures
          - 'simulate_tournament'    → run Monte Carlo simulation
          - 'load_models'            → load pre-trained models from disk
        """
        print(f"[PredictionAgent] Task: {task}")
        if task == "train":
            return self._train(**kwargs)
        elif task == "predict_match":
            return self._predict_match(**kwargs)
        elif task == "predict_all_fixtures":
            return self._predict_all_fixtures()
        elif task == "simulate_tournament":
            return self._simulate_tournament(**kwargs)
        elif task == "load_models":
            return self._load_models()
        return {"error": f"Unknown task: {task}"}

    # ── Training ──────────────────────────────────────────────────────────
    def _train(self, df: Optional[pd.DataFrame] = None) -> dict:
        if df is None:
            df = pd.read_csv(DATA_DIR / "match_features.csv", parse_dates=["date"])

        X = df[FEATURE_COLS].fillna(0.5)
        y = df["team1_won"]

        # Time-based split: last 2 seasons = test
        train_mask = df["season"].apply(
            lambda s: int(str(s)[:4]) < 2024
        )
        X_tr, X_te = X[train_mask], X[~train_mask]
        y_tr, y_te = y[train_mask], y[~train_mask]
        print(f"[PredictionAgent] Train: {len(X_tr)} | Test: {len(X_te)}")

        # ── XGBoost ──
        xgb_m = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
        xgb_m.fit(X_tr, y_tr)

        # ── LightGBM ──
        lgb_m = lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        lgb_m.fit(X_tr, y_tr)

        # ── Random Forest ──
        rf_m = RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=42
        )
        rf_m.fit(X_tr, y_tr)

        # ── Evaluate ──
        metrics = {}
        for name, model in [("xgb", xgb_m), ("lgb", lgb_m), ("rf", rf_m)]:
            prob = model.predict_proba(X_te)[:, 1]
            pred = (prob > 0.5).astype(int)
            metrics[name] = {
                "accuracy": round(accuracy_score(y_te, pred), 4),
                "auc": round(roc_auc_score(y_te, prob), 4),
            }
            print(f"  {name}: accuracy={metrics[name]['accuracy']:.3f}  AUC={metrics[name]['auc']:.3f}")

        # Ensemble
        ens_prob = (
            xgb_m.predict_proba(X_te)[:, 1]
            + lgb_m.predict_proba(X_te)[:, 1]
            + rf_m.predict_proba(X_te)[:, 1]
        ) / 3
        metrics["ensemble"] = {
            "accuracy": round(accuracy_score(y_te, (ens_prob > 0.5).astype(int)), 4),
            "auc": round(roc_auc_score(y_te, ens_prob), 4),
        }
        print(f"  ensemble: accuracy={metrics['ensemble']['accuracy']:.3f}  AUC={metrics['ensemble']['auc']:.3f}")

        self.models = {"xgb": xgb_m, "lgb": lgb_m, "rf": rf_m}
        self._save_models()
        return {"metrics": metrics}

    # ── Single Match Prediction ───────────────────────────────────────────
    def _predict_match(self, team1: str, team2: str, features: Optional[dict] = None,
                       venue: str = "", toss_winner: str = "",
                       toss_decision: str = "field") -> dict:
        if self.models is None:
            self._load_models()

        if features is None:
            from feature_agent import FeatureAgent
            fa = FeatureAgent().load()
            X = fa.get_match_features(team1, team2, venue, toss_winner, toss_decision)
        else:
            X = pd.DataFrame([features])[FEATURE_COLS]

        X = X.fillna(0.5)
        xp = self.models["xgb"].predict_proba(X)[0, 1]
        lp = self.models["lgb"].predict_proba(X)[0, 1]
        rp = self.models["rf"].predict_proba(X)[0, 1]
        p1 = (xp + lp + rp) / 3

        return {
            "team1": team1,
            "team2": team2,
            "team1_win_prob": round(p1, 4),
            "team2_win_prob": round(1 - p1, 4),
            "predicted_winner": team1 if p1 > 0.5 else team2,
            "confidence": round(max(p1, 1 - p1), 4),
            "model_breakdown": {"xgb": round(xp, 4), "lgb": round(lp, 4), "rf": round(rp, 4)},
        }

    # ── Predict All Fixtures ──────────────────────────────────────────────
    def _predict_all_fixtures(self) -> dict:
        if self.models is None:
            self._load_models()

        fixtures = pd.read_csv(DATA_DIR / "ipl_2026_fixtures.csv")
        results = []
        for _, row in fixtures.iterrows():
            pred = self._predict_match(
                team1=row["home_team"], team2=row["away_team"],
                venue=str(row.get("venue", "")),
            )
            pred["match_num"] = row["match_num"]
            pred["date"] = row["date"]
            results.append(pred)

        df_out = pd.DataFrame(results)
        out_path = DATA_DIR / "ipl_2026_predictions.csv"
        df_out.to_csv(out_path, index=False)
        print(f"[PredictionAgent] Saved {len(df_out)} predictions → {out_path}")
        return {"predictions": results, "path": str(out_path)}

    # ── Monte Carlo Tournament Simulation ────────────────────────────────
    def _simulate_tournament(self, n: int = 50000) -> dict:
        """Run n Monte Carlo simulations of IPL 2026 tournament."""
        if self.models is None:
            self._load_models()

        # Pre-compute all pairwise win probabilities
        print("[PredictionAgent] Pre-computing pairwise probabilities...")
        wp = self._build_win_prob_table()

        print(f"[PredictionAgent] Running {n:,} simulations...")
        random.seed(42)

        champ_count = {t: 0 for t in IPL_2026_TEAMS}
        playoff_count = {t: 0 for t in IPL_2026_TEAMS}
        top2_count = {t: 0 for t in IPL_2026_TEAMS}

        for _ in range(n):
            champ, wins = self._single_sim(wp)
            champ_count[champ] += 1
            sorted_t = sorted(wins.items(), key=lambda x: (-x[1], x[0]))
            for t, _ in sorted_t[:4]:
                playoff_count[t] += 1
            for t, _ in sorted_t[:2]:
                top2_count[t] += 1

        results = {}
        for t in IPL_2026_TEAMS:
            results[t] = {
                "win_probability": round(champ_count[t] / n, 4),
                "playoff_probability": round(playoff_count[t] / n, 4),
                "top2_probability": round(top2_count[t] / n, 4),
            }

        # Sort by win probability
        results = dict(sorted(results.items(), key=lambda x: -x[1]["win_probability"]))

        out_path = DATA_DIR / "monte_carlo_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[PredictionAgent] Monte Carlo results saved → {out_path}")

        return {"simulations": n, "results": results, "path": str(out_path)}

    def _single_sim(self, wp: Dict) -> Tuple[str, Dict]:
        """Run a single tournament simulation."""
        wins = {t: 0 for t in IPL_2026_TEAMS}

        # League stage: approximate double round-robin
        for t1 in IPL_2026_TEAMS:
            for t2 in IPL_2026_TEAMS:
                if t1 < t2:
                    for _ in range(2):
                        if random.random() < wp[(t1, t2)]:
                            wins[t1] += 1
                        else:
                            wins[t2] += 1

        sorted_t = sorted(wins.items(), key=lambda x: (-x[1], x[0]))
        top4 = [t[0] for t in sorted_t[:4]]

        # Q1: 1st vs 2nd → winner goes to final, loser to Q2
        q1_w = top4[0] if random.random() < wp[(min(top4[0], top4[1]), max(top4[0], top4[1]))] == wp.get((top4[0], top4[1]), 0.5) else top4[1]
        q1_w = top4[0] if random.random() < self._get_wp(wp, top4[0], top4[1]) else top4[1]
        q1_l = top4[1] if q1_w == top4[0] else top4[0]

        # Eliminator: 3rd vs 4th
        elim_w = top4[2] if random.random() < self._get_wp(wp, top4[2], top4[3]) else top4[3]

        # Q2: Q1 loser vs Eliminator winner
        q2_w = q1_l if random.random() < self._get_wp(wp, q1_l, elim_w) else elim_w

        # Final
        champ = q1_w if random.random() < self._get_wp(wp, q1_w, q2_w) else q2_w

        return champ, wins

    def _build_win_prob_table(self) -> Dict:
        """Compute pairwise win probabilities for all team pairs."""
        from feature_agent import FeatureAgent
        fa = FeatureAgent().load()
        wp = {}
        for t1 in IPL_2026_TEAMS:
            for t2 in IPL_2026_TEAMS:
                if t1 != t2:
                    X = fa.get_match_features(t1, t2)
                    xp = self.models["xgb"].predict_proba(X)[0, 1]
                    lp = self.models["lgb"].predict_proba(X)[0, 1]
                    rp = self.models["rf"].predict_proba(X)[0, 1]
                    wp[(t1, t2)] = (xp + lp + rp) / 3
        return wp

    @staticmethod
    def _get_wp(wp: Dict, t1: str, t2: str) -> float:
        return wp.get((t1, t2), 1 - wp.get((t2, t1), 0.5))

    # ── Model Persistence ─────────────────────────────────────────────────
    def _save_models(self):
        path = DATA_DIR / "models.pkl"
        with open(path, "wb") as f:
            pickle.dump({"models": self.models, "feature_cols": FEATURE_COLS}, f)
        print(f"[PredictionAgent] Models saved → {path}")

    def _load_models(self) -> dict:
        path = DATA_DIR / "models.pkl"
        if not path.exists():
            return {"error": "models.pkl not found. Run train first."}
        with open(path, "rb") as f:
            saved = pickle.load(f)
        self.models = saved["models"]
        print("[PredictionAgent] Models loaded from disk.")
        return {"status": "loaded"}


if __name__ == "__main__":
    agent = PredictionAgent()
    print("Training models...")
    agent.run("train")
    print("\nPredicting Match 1: RCB vs SRH")
    result = agent.run("predict_match", team1="Royal Challengers Bengaluru",
                       team2="Sunrisers Hyderabad", venue="M Chinnaswamy Stadium")
    print(result)
