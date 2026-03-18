"""
IPL Prediction System - Orchestrator Agent
===========================================
The master coordinator. Manages all sub-agents and runs the full pipeline:

  1. DataAgent     → fetch/update fixtures & historical data
  2. FeatureAgent  → engineer features from raw data
  3. PredictionAgent → train models, predict matches, simulate tournament

Usage:
  python orchestrator_agent.py --task full_pipeline
  python orchestrator_agent.py --task predict_all
  python orchestrator_agent.py --task simulate
  python orchestrator_agent.py --task update         # after new results
"""

import sys, json, argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add agent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_agent import DataAgent
from feature_agent import FeatureAgent
from prediction_agent import PredictionAgent

DATA_DIR = Path(__file__).parent.parent / "data"


class OrchestratorAgent:
    """
    Coordinates the full IPL prediction pipeline.

    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │              Orchestrator Agent                      │
    │  (pipeline control, reporting, state management)    │
    └────────┬──────────────┬──────────────┬──────────────┘
             │              │              │
        ┌────▼────┐  ┌──────▼──────┐  ┌───▼──────────┐
        │  Data   │  │  Feature    │  │  Prediction  │
        │  Agent  │  │  Agent      │  │  Agent       │
        │fetch,   │  │feature eng, │  │XGB+LGB+RF,   │
        │clean,   │  │rolling stats│  │Monte Carlo   │
        │update   │  │H2H, venue   │  │50k sims      │
        └─────────┘  └─────────────┘  └──────────────┘
    """

    def __init__(self):
        self.data_agent = DataAgent()
        self.feature_agent = FeatureAgent()
        self.pred_agent = PredictionAgent()
        self.state = {}

    def run(self, task: str) -> dict:
        """
        Top-level task dispatcher:
          - 'full_pipeline'    → run everything from scratch
          - 'predict_all'      → predict all 2026 fixtures (requires trained models)
          - 'simulate'         → run Monte Carlo tournament simulation
          - 'update'           → refresh data after new match results
          - 'report'           → print summary report
        """
        print(f"\n{'='*60}")
        print(f"[Orchestrator] Starting task: {task}")
        print(f"{'='*60}\n")

        if task == "full_pipeline":
            return self._full_pipeline()
        elif task == "predict_all":
            return self._predict_all()
        elif task == "simulate":
            return self._simulate()
        elif task == "update":
            return self._update_after_match()
        elif task == "report":
            return self._generate_report()
        else:
            return {"error": f"Unknown task: {task}"}

    # ── Full Pipeline ─────────────────────────────────────────────────────
    def _full_pipeline(self) -> dict:
        """Run the complete pipeline from scratch."""
        results = {}

        # Step 1: Fetch data
        print("\n[Step 1/5] Fetching IPL 2026 fixtures...")
        r = self.data_agent.run("fetch_fixtures")
        results["fixtures"] = r
        print(f"  → {r.get('matches', '?')} fixtures downloaded")

        # Step 2: Load historical data (already processed)
        print("\n[Step 2/5] Loading historical match data...")
        r = self.data_agent.run("load_matches")
        df_hist = r.get("dataframe")
        if df_hist is None:
            print("  ⚠ Historical data not found. Fetching from CricSheet...")
            self.data_agent.run("fetch_historical")
            r = self.data_agent.run("load_matches")
            df_hist = r.get("dataframe")
        print(f"  → {r.get('rows', '?')} historical matches loaded")
        results["historical"] = {"rows": r.get("rows")}

        # Step 3: Build features
        print("\n[Step 3/5] Engineering features...")
        self.feature_agent.load(df_hist)
        r = self.feature_agent.run("build_matrix")
        df_feat = r.get("dataframe")
        print(f"  → Feature matrix: {r.get('rows', '?')} × {len(df_feat.columns) if df_feat is not None else '?'} columns")
        results["features"] = {"rows": r.get("rows")}

        # Step 4: Train models
        print("\n[Step 4/5] Training ensemble model (XGBoost + LightGBM + RF)...")
        r = self.pred_agent.run("train", df=df_feat)
        metrics = r.get("metrics", {})
        ens = metrics.get("ensemble", {})
        print(f"  → Ensemble | Accuracy: {ens.get('accuracy', '?'):.3f} | AUC: {ens.get('auc', '?'):.3f}")
        results["training"] = metrics

        # Step 5: Predict & simulate
        print("\n[Step 5/5] Generating predictions & running tournament simulation...")
        pred_r = self.pred_agent.run("predict_all_fixtures")
        sim_r = self.pred_agent.run("simulate_tournament", n=50000)
        results["predictions"] = {"count": len(pred_r.get("predictions", []))}
        results["simulation"] = {"simulations": sim_r.get("simulations")}

        print("\n✅ Full pipeline complete!")
        self._print_summary(sim_r)
        return results

    # ── Predict All ───────────────────────────────────────────────────────
    def _predict_all(self) -> dict:
        self.pred_agent.run("load_models")
        r = self.pred_agent.run("predict_all_fixtures")
        self._print_fixture_predictions(r.get("predictions", []))
        return r

    # ── Simulate ─────────────────────────────────────────────────────────
    def _simulate(self) -> dict:
        self.pred_agent.run("load_models")
        r = self.pred_agent.run("simulate_tournament", n=50000)
        self._print_summary(r)
        return r

    # ── Update After Match ────────────────────────────────────────────────
    def _update_after_match(self) -> dict:
        """
        Call after new match results are available.
        Refreshes fixtures, re-runs predictions.
        """
        print("[Orchestrator] Updating with latest results...")
        r1 = self.data_agent.run("update_results")
        self.pred_agent.run("load_models")
        r2 = self.pred_agent.run("predict_all_fixtures")
        r3 = self.pred_agent.run("simulate_tournament", n=50000)
        self._print_summary(r3)
        return {"fixtures_updated": r1, "predictions": r2, "simulation": r3}

    # ── Report ────────────────────────────────────────────────────────────
    def _generate_report(self) -> dict:
        preds_path = DATA_DIR / "ipl_2026_predictions.csv"
        mc_path = DATA_DIR / "monte_carlo_results.json"

        if preds_path.exists():
            df = pd.read_csv(preds_path)
            self._print_fixture_predictions(df.to_dict("records"))

        if mc_path.exists():
            with open(mc_path) as f:
                mc = json.load(f)
            self._print_summary({"results": mc})

        return {"status": "report generated"}

    # ── Pretty Printers ───────────────────────────────────────────────────
    @staticmethod
    def _print_fixture_predictions(predictions: list):
        print("\n" + "="*90)
        print(f"{'IPL 2026 MATCH PREDICTIONS':^90}")
        print("="*90)
        for p in predictions:
            t1 = p.get("home_team", p.get("team1", ""))
            t2 = p.get("away_team", p.get("team2", ""))
            p1 = p.get("home_win_prob", p.get("team1_win_prob", 0.5))
            p2 = p.get("away_win_prob", p.get("team2_win_prob", 0.5))
            winner = p.get("predicted_winner", t1 if p1 > p2 else t2)
            date = p.get("date", "")
            match = p.get("match_num", "")
            print(f"  {str(match):<10} {str(date):<12} | "
                  f"{t1:<35} ({float(p1):.0%}) vs "
                  f"{t2:<35} ({float(p2):.0%}) → 🏆 {winner}")

    @staticmethod
    def _print_summary(sim_result: dict):
        results = sim_result.get("results", {})
        n = sim_result.get("simulations", sim_result.get("n", 50000))
        print("\n" + "="*75)
        print(f"{'IPL 2026 CHAMPIONSHIP PREDICTIONS':^75}")
        print(f"{'Monte Carlo Simulation (' + str(n) + ' runs)':^75}")
        print("="*75)
        print(f"{'Team':<35} {'Win %':>8}  {'Playoff %':>10}  Chart")
        print("-"*75)
        for team, stats in results.items():
            wp = stats.get("win_probability", 0)
            pp = stats.get("playoff_probability", 0)
            bar = "█" * int(wp * 100 / 2.5)
            print(f"{team:<35} {wp*100:>7.1f}%  {pp*100:>9.1f}%  {bar}")
        print("="*75)


# ── CLI Entry Point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPL 2026 Prediction Orchestrator")
    parser.add_argument(
        "--task",
        default="full_pipeline",
        choices=["full_pipeline", "predict_all", "simulate", "update", "report"],
        help="Task to run",
    )
    args = parser.parse_args()

    orchestrator = OrchestratorAgent()
    result = orchestrator.run(args.task)
    print("\nDone.")
