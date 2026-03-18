"""
IPL Prediction System - Data Agent
====================================
Responsible for:
  - Fetching IPL 2026 fixtures from IPLT20 stats API
  - Downloading historical match data from CricSheet
  - Updating match results after each game day
  - Normalizing team names across seasons
"""

import os, re, json, requests, zipfile, io
import pandas as pd
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

IPL_2026_TEAMS = [
    "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans",
    "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
    "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
}


class DataAgent:
    """Fetches, cleans, and updates all IPL data."""

    def __init__(self):
        self.data_dir = DATA_DIR

    # ── Public API ────────────────────────────────────────────────────────
    def run(self, task: str) -> dict:
        """
        Dispatch tasks:
          - 'fetch_fixtures'      → download IPL 2026 schedule
          - 'fetch_historical'    → download CricSheet ball-by-ball data
          - 'update_results'      → refresh fixtures with latest scores
          - 'load_matches'        → return clean match dataframe
        """
        print(f"[DataAgent] Task: {task}")
        if task == "fetch_fixtures":
            return self._fetch_fixtures()
        elif task == "fetch_historical":
            return self._fetch_historical()
        elif task == "update_results":
            return self._update_results()
        elif task == "load_matches":
            return self._load_matches()
        else:
            return {"error": f"Unknown task: {task}"}

    # ── Fetch IPL 2026 Fixtures ───────────────────────────────────────────
    def _fetch_fixtures(self) -> dict:
        """Download IPL 2026 fixture list from IPLT20 stats feed."""
        # CompID 284 = IPL 2026 (discovered via IPLT20 stats S3 bucket)
        url = ("https://ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com"
               "/ipl/feeds/284-matchschedule.js")
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
        except Exception as e:
            return {"error": str(e), "source": url}

        content = r.text
        m = re.search(r"MatchSchedule\((.*)\)", content, re.DOTALL)
        if not m:
            return {"error": "Could not parse JSONP response"}

        data = json.loads(m.group(1))
        matches = data["Matchsummary"]

        rows = []
        for match in matches:
            rows.append({
                "match_num":       match.get("MatchOrder", ""),
                "match_id":        match.get("MatchID", ""),
                "date":            match.get("MatchDate", ""),
                "time_ist":        match.get("MatchTime", ""),
                "home_team":       self._normalize(match.get("HomeTeamName", "")),
                "away_team":       self._normalize(match.get("AwayTeamName", "")),
                "team1":           self._normalize(match.get("FirstBattingTeamName", "")),
                "team2":           self._normalize(match.get("SecondBattingTeamName", "")),
                "venue":           match.get("GroundName", ""),
                "city":            match.get("city", ""),
                "status":          match.get("MatchStatus", ""),
                "result":          match.get("Comments", "") or match.get("Commentss", ""),
                "winner_id":       match.get("WinningTeamID", ""),
                "toss_winner":     self._normalize(match.get("TossTeam", "")),
                "toss_details":    match.get("TossDetails", ""),
            })

        df = pd.DataFrame(rows)
        out_path = self.data_dir / "ipl_2026_fixtures.csv"
        df.to_csv(out_path, index=False)
        print(f"[DataAgent] Saved {len(df)} fixtures → {out_path}")
        return {"matches": len(df), "path": str(out_path)}

    # ── Fetch Historical Data ─────────────────────────────────────────────
    def _fetch_historical(self) -> dict:
        """Download and parse CricSheet IPL ball-by-ball CSVs."""
        zip_url = "https://cricsheet.org/downloads/ipl_male_csv2.zip"
        print(f"[DataAgent] Downloading {zip_url} ...")
        try:
            r = requests.get(zip_url, headers=HEADERS, timeout=120, stream=True)
            r.raise_for_status()
            content = r.content
        except Exception as e:
            return {"error": str(e)}

        zip_dir = self.data_dir / "cricsheet_raw"
        zip_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zf.extractall(zip_dir)

        # Parse info files → match-level dataframe
        match_records = []
        for f in sorted(zip_dir.iterdir()):
            if not f.name.endswith("_info.csv"):
                continue
            rec = {"match_id": f.name.replace("_info.csv", "")}
            teams, players = [], {}
            with open(f) as fp:
                for line in fp:
                    parts = line.strip().split(",", 2)
                    if len(parts) < 3 or parts[0] != "info":
                        continue
                    key, val = parts[1], parts[2].strip('"')
                    if key == "player":
                        tp = val.split(",", 1)
                        if len(tp) == 2:
                            players.setdefault(tp[0].strip(), []).append(tp[1].strip())
                    elif key == "team":
                        teams.append(val)
                    else:
                        rec[key] = val
            if len(teams) >= 2:
                rec["team1"], rec["team2"] = self._normalize(teams[0]), self._normalize(teams[1])
            match_records.append(rec)

        df = pd.DataFrame(match_records)
        # Normalize columns
        for col in ["winner", "toss_winner"]:
            if col in df.columns:
                df[col] = df[col].apply(self._normalize)

        out_path = self.data_dir / "matches_historical.csv"
        df.to_csv(out_path, index=False)
        print(f"[DataAgent] Parsed {len(df)} historical matches → {out_path}")
        return {"matches": len(df), "path": str(out_path)}

    # ── Update Results ────────────────────────────────────────────────────
    def _update_results(self) -> dict:
        """Re-fetch fixtures to get latest results (call after each match day)."""
        return self._fetch_fixtures()

    # ── Load Matches ──────────────────────────────────────────────────────
    def _load_matches(self) -> dict:
        path = self.data_dir / "matches_clean.csv"
        if not path.exists():
            return {"error": "matches_clean.csv not found. Run fetch_historical first."}
        df = pd.read_csv(path, parse_dates=["date"])
        return {"dataframe": df, "rows": len(df)}

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _normalize(name: str) -> str:
        return TEAM_MAP.get(name.strip(), name.strip())


if __name__ == "__main__":
    agent = DataAgent()
    print("\n=== Fetching IPL 2026 Fixtures ===")
    result = agent.run("fetch_fixtures")
    print(result)
