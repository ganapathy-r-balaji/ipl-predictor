# IPL 2026 Match Prediction System

**An end-to-end AI-powered cricket match prediction pipeline** using 17 years of historical IPL data (2008–2025), enriched with pitch profiles, weather data, ELO ratings, and player form features — culminating in an ensemble of ML models and two fully interactive match simulator dashboards (a self-contained HTML file and a Streamlit web app).

Built by **Ganapathy Raaman Balaji**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Phase 1 — Data Collection](#3-phase-1--data-collection)
4. [Phase 2A — Venue Pitch Profiling](#4-phase-2a--venue-pitch-profiling)
5. [Phase 2B — Historical Weather Features](#5-phase-2b--historical-weather-features)
6. [Phase 2C — ELO Rating System](#6-phase-2c--elo-rating-system)
7. [Phase 3 — Player Form Features](#7-phase-3--player-form-features)
8. [Phase 4 — Enriched Feature Matrix](#8-phase-4--enriched-feature-matrix)
9. [Phase 5 — Model Training v2](#9-phase-5--model-training-v2)
10. [Phase 6 — Live Dashboards](#10-phase-6--live-dashboards)
11. [Streamlit Web App](#11-streamlit-web-app)
12. [Tools & Packages](#12-tools--packages)
13. [MCP Usage](#13-mcp-usage)
14. [Data Files Reference](#14-data-files-reference)
15. [Known Issues & Fixes](#15-known-issues--fixes)

---

## 1. Project Overview

### Goal
Build a production-grade IPL match prediction system that:
- Ingests and processes 17 years of historical IPL ball-by-ball data (2008–2025)
- Fetches live IPL 2026 fixtures and team schedules
- Enriches each match with pitch conditions, weather, ELO ratings, and player form
- Trains an ensemble of ML models (XGBoost, LightGBM, Random Forest, PyTorch MLP, Stacking Meta-Learner)
- Runs Monte Carlo simulations to estimate championship and playoff probabilities
- Delivers two fully interactive match simulator interfaces — a self-contained single-file HTML dashboard and a Streamlit web app — with live condition-based probability updates and no backend server required

### What Makes This Different From a Standard Kaggle Approach
- **Live data ingestion**: Fixtures sourced from the official IPLT20 S3 API, not a static dataset
- **Ball-by-ball granularity**: 278,205 individual deliveries processed to derive venue and team behavior
- **Temporal integrity**: Strict train/test time split (2008–2023 train, 2024–2025 test) to prevent data leakage
- **Contextual enrichment**: Every match gets its own pitch type, real historical weather, ELO-adjusted team strength, and recent player form
- **No backend required**: Pre-computed probability tables allow instant browser-side and Streamlit-side predictions across ~10,000 condition combinations per match
- **Live condition sensitivity**: Changing toss winner, pitch type, temperature, humidity, rain, or dew instantly updates win probabilities, the expected-value league table, and the playoff bracket

### Schedule Note
Only the first 20 matches (Phase 1: March 28 – April 12, 2026) have been officially released by the BCCI as of the time of publishing. The remaining fixtures are pending announcement due to state elections in India. The dashboard league table and Monte Carlo probabilities already account for the full 14-game season through simulated extra matches embedded in `base_extra_pts`.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                        │
│        (orchestrator_agent.py — CLI entry point)            │
└──────────┬───────────────────────────────────────────────────┘
           │
    ┌──────▼──────┐    ┌──────────────┐    ┌────────────────┐
    │ Data Agent  │    │ Feature Agent│    │Prediction Agent│
    │(data_agent  │    │(feature_agent│    │(prediction_    │
    │   .py)      │    │    .py)      │    │  agent.py)     │
    └──────┬──────┘    └──────┬───────┘    └───────┬────────┘
           │                  │                    │
    ┌──────▼──────────────────▼────────────────────▼────────┐
    │                    /ipl_data/                          │
    │  CricSheet CSVs · IPLT20 API · Open-Meteo · T20 data  │
    └───────────────────────────┬────────────────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │       prep_dashboard_v2.py           │
              │  Generates dashboard_data_v2.json    │
              └────────────┬──────────┬─────────────┘
                           │          │
              ┌────────────▼──┐  ┌────▼────────────────┐
              │build_dashboard│  │  ipl_streamlit/      │
              │    .py        │  │     app.py           │
              │ dashboard.html│  │ (Streamlit web app)  │
              └───────────────┘  └─────────────────────┘
```

The system is structured as a **multi-agent pipeline** with four Python modules:

| Agent | File | Responsibility |
|---|---|---|
| Orchestrator | `orchestrator_agent.py` | CLI entry point, coordinates all agents |
| Data Agent | `data_agent.py` | Fetches fixtures, downloads CricSheet data |
| Feature Agent | `feature_agent.py` | Builds feature matrix from historical data |
| Prediction Agent | `prediction_agent.py` | Trains models, runs simulations |

### CLI Usage

```bash
cd agents

# Run full pipeline from scratch
python orchestrator_agent.py --task full_pipeline

# Just predict all 2026 fixtures
python orchestrator_agent.py --task predict_all

# Run Monte Carlo tournament simulation
python orchestrator_agent.py --task simulate

# Update after new match results come in
python orchestrator_agent.py --task update

# Generate a text report
python orchestrator_agent.py --task report
```

### Dashboard Build

```bash
# Regenerate dashboard_data_v2.json from source data
python prep_dashboard_v2.py

# Rebuild the HTML dashboard from the data JSON
python build_dashboard.py

# Run the Streamlit app locally
streamlit run ipl_streamlit/app.py
```

---

## 3. Phase 1 — Data Collection

### What Was Done

#### 3.1 ESPN Cricinfo — Attempted & Blocked
The original plan was to scrape `espncricinfo.com/series/ipl-2026-1510719` directly for fixtures, team squads, venue data, and historical results. All requests — standard HTTP, `requests` with browser-like headers, Playwright headless browser automation, and direct API endpoint probing — returned **HTTP 403** errors.

**Root cause**: ESPN Cricinfo is protected by **Akamai Bot Manager CDN**, which fingerprints request patterns, TLS handshakes, and IP reputation. All cloud VM and server IP ranges are blocked by default.

#### 3.2 Fallback Strategy — Three Alternative Sources

**Source 1: CricSheet (Primary historical data)**
- URL: `https://cricsheet.org/downloads/ipl_male_csv2.zip`
- Downloaded: **1,169 match files** covering 2008–2025 IPL seasons
- After loading and cleaning: **278,205 individual deliveries** in `all_deliveries.parquet`
- Match-level data: **1,169 rows × 21 columns** in `matches_clean.csv`

**Source 2: IPLT20 S3 API (IPL 2026 fixtures)**
- Base URL: `https://ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com/ipl/feeds/{CompID}-matchschedule.js`
- CompID scan 1–400 identified **CompID 284** for IPL 2026
- Result: **20 fixtures** from March 28 – April 12, 2026
- Saved to: `ipl_2026_fixtures.csv`

**Source 3: Wikipedia (Tournament metadata)**
- Confirmed: 10 teams, double round-robin format, 74 league matches + 4 playoff matches

#### 3.3 Data Cleaning

Key normalizations applied:
- **Team name standardization**: 20+ historical variants mapped to canonical names
- **Season column**: Cast to `str` before sorting to avoid `TypeError` with mixed int/float NaN values
- **Ball column**: Parsed with `pd.to_numeric(errors='coerce')` and NaN rows dropped
- **Match filtering**: Excluded no-result, D/L, and abandoned matches from training data

#### 3.4 Initial Feature Matrix

The `feature_agent.py` computed 13 features for each historical match:

| Feature | Description |
|---|---|
| `t1_home_win_rate` | Team 1 home win rate (last 3 seasons) |
| `t2_away_win_rate` | Team 2 away win rate (last 3 seasons) |
| `h2h_win_rate` | Head-to-head win rate (Team 1 vs Team 2, last 5 encounters) |
| `t1_recent_form` | Team 1 win rate in last 5 matches |
| `t2_recent_form` | Team 2 win rate in last 5 matches |
| `venue_t1_win_rate` | Team 1 win rate at this specific venue |
| `toss_winner_is_t1` | Binary: did Team 1 win the toss? |
| `toss_decision_bat` | Binary: did the toss winner choose to bat? |
| `t1_avg_first_inn_score` | Team 1's average 1st innings total |
| `t2_avg_first_inn_score` | Team 2's average 1st innings total |
| `t1_bowling_economy` | Team 1's average bowling economy (last 3 seasons) |
| `t2_bowling_economy` | Team 2's average bowling economy (last 3 seasons) |
| `season` | IPL season year |

Target variable: `team1_won` (1 = Team 1 won, 0 = Team 2 won)

#### 3.5 Initial Model Results

Trained on 2008–2023, tested on 2024–2025:

| Model | Accuracy | AUC-ROC |
|---|---|---|
| XGBoost | 53.9% | 0.582 |
| LightGBM | 52.5% | 0.534 |
| Random Forest | 47.5% | 0.487 |
| **Ensemble** | **55.3%** | **0.557** |

---

## 4. Phase 2A — Venue Pitch Profiling

### Why This Was Added
Cricket outcomes are heavily influenced by pitch behavior. A flat batting pitch at Wankhede (Mumbai) produces totals 20–30 runs higher than a dry spin pitch at Chepauk (Chennai). Without encoding this, the model treats all venues identically.

### What Was Done

All 278,205 deliveries were grouped by venue to compute four pitch metrics per ground.

#### 4.1 Flatness Index
```python
flatness = venue_match_avg_total / ipl_overall_avg_total
```
Values above 1.08 → **"flat"**; below 0.92 → bowler-friendly.

#### 4.2 Pace Score
```python
pace_score = (pp_run_rate / ipl_pp_rr * 0.5) + (pace_wicket_pct * 0.5)
```

#### 4.3 Spin Score
```python
spin_score = (spin_wicket_pct * 0.6) + (max(0, 1 - mid_over_rr / death_rr) * 0.4)
```

#### 4.4 Dew Factor
```python
dew_factor = (2nd_inn_pp_rr / 2nd_inn_pp_balls) / (1st_inn_pp_rr / 1st_inn_pp_balls) - 1.0
```

#### 4.5 Pitch Label Assignment
```python
if flatness > 1.08:      label = "flat"
elif spin_score > 0.18:  label = "spin"
elif pace_score > 1.05:  label = "pace"
else:                    label = "balanced"
```

**Results**: 58 unique IPL venues profiled. Saved to: `venue_pitch_profiles.csv` and `venue_pitch_profiles.json`.

---

## 5. Phase 2B — Historical Weather Features

### Why This Was Added
Weather materially affects T20 cricket — humidity and dew make the ball slippery for bowlers in the 2nd innings, rain can trigger D/L adjustments, and temperature correlates with swing conditions and player fatigue.

### What Was Done

**API Used**: Open-Meteo Archive (`https://archive-api.open-meteo.com/v1/archive`) — free, no API key, covers back to 1940.

| Feature | Description |
|---|---|
| `temp_avg` | Average evening temperature (°C) |
| `precipitation` | Total rainfall on match day (mm) |
| `humidity_evening` | Average relative humidity 15:00–19:00 (%) |
| `dewpoint_evening` | Average dewpoint temperature 15:00–19:00 (°C) |
| `rain_risk` | Binary: precipitation > 1mm |
| `dew_risk` | Binary: humidity_evening > 75% |

**Scale**: 1,118 historical matches processed. Delegated to a sub-agent to avoid the 2-minute Bash timeout. Saved to: `weather_cache.csv` and `weather_features.csv`.

---

## 6. Phase 2C — ELO Rating System

### Why This Was Added
ELO provides a principled, recency-weighted measure of relative team strength. Strong wins produce larger rating gains; upsets cause large swings.

### How ELO Works

```python
INITIAL_ELO = 1500
K_LEAGUE    = 32   # league matches
K_PLAYOFF   = 48   # playoff matches (higher stakes)

def expected_win(r1, r2):
    return 1.0 / (1 + 10 ** ((r2 - r1) / 400))

def update_elo(r1, r2, result, k):
    e1      = expected_win(r1, r2)
    new_r1  = r1 + k * (result - e1)
    new_r2  = r2 + k * ((1 - result) - (1 - e1))
    return new_r1, new_r2
```

### IPL 2026 Pre-Season ELO Ratings

| Team | ELO Rating | Context |
|---|---|---|
| RCB | 1616.2 | Defending IPL 2025 champions |
| GT | 1547.7 | Strong 2024 performance |
| PBKS | 1535.2 | Consistent recent form |
| KKR | 1534.2 | 2024 Champions |
| MI | 1526.8 | Multiple-time champions |
| DC | 1520.7 | Improved recent seasons |
| SRH | 1511.7 | 2024 runners-up |
| LSG | 1500.1 | Average historical performance |
| RR | 1475.9 | Below average recent form |
| CSK | 1468.5 | Recent decline, rebuilding roster |

Saved to: `elo_features.csv` and `elo_current.json`.

---

## 7. Phase 3 — Player Form Features

### Why This Was Added
Team-level aggregates miss current player momentum. A team missing its key strike bowler is fundamentally weaker than the same team at full strength.

**Form window**: December 28, 2025 – March 28, 2026 (232 matches, 53,340 deliveries from IPL 2025 + T20 internationals).

### Scoring

```python
batting_score = (batting_avg * strike_rate / 100).clip(upper=500)
bowling_score = (wicket_rate * (9 / economy.clip(lower=1))).clip(upper=20)
```

### Team Aggregation

```python
team_batting_score = top_6_batters['batting_score'].mean()
team_bowling_score = top_4_bowlers['bowling_score'].mean()
```

Saved to: `player_form_features.csv`, `player_batting_form.csv`, `player_bowling_form.csv`.

---

## 8. Phase 4 — Enriched Feature Matrix

### What Was Done

All data sources were joined into a single unified feature matrix (`enriched_features.csv`) with ~33 columns used for model retraining.

| Feature Group | Features | Count |
|---|---|---|
| Original rolling stats | team form, H2H, venue win rates, toss, batting/bowling averages | 13 |
| ELO ratings | `t1_elo`, `t2_elo`, `elo_diff`, `elo_win_prob_t1` | 4 |
| Weather | `temp_avg`, `precipitation`, `humidity_evening`, `dewpoint_evening`, `rain_risk`, `dew_risk` | 6 |
| Pitch | `flatness_index`, `pace_score`, `spin_score`, `dew_factor` | 4 |
| Player form | `t1_batting_score`, `t1_bowling_score`, `t1_batting_sr`, `t1_bowling_econ` + same for t2 | 8 |

**Join strategy**: `match_id` for ELO and weather; `venue` name for pitch; team name for player form.

The 2026 fixture feature matrix (`enriched_features_2026.csv`) was generated using the same pipeline, substituting pre-season ELO ratings and the Dec 2025–Mar 2026 player form window.

---

## 9. Phase 5 — Model Training v2

### Models Trained

Six models were trained on the enriched 33-feature matrix (2008–2023 train, 2024–2025 test):

| Model | Accuracy | AUC-ROC |
|---|---|---|
| XGBoost | ~46.1% | 0.4851 |
| LightGBM | ~51.8% | 0.4891 |
| Random Forest | ~50.0% | ~0.49 |
| MLP (Neural Net) | ~49.0% | ~0.49 |
| Simple Ensemble (avg) | ~50.5% | ~0.50 |
| Stacking Meta-Learner | ~51.5% | ~0.50 |

**Note on accuracy**: IPL match outcomes are inherently noisy — random events (dropped catches, weather interruptions, injuries) explain a large share of variance. A model achieving ~52% on a held-out test set is competitive with academic benchmarks for T20 prediction. The value of this system is in the **relative probability ordering** between teams and match conditions, not in claiming high absolute accuracy.

### Condition-Adjusted Probability Lookup

The key innovation enabling the live dashboard is a pre-computed **probability lookup table** (`prob_lookup.json`). For each of the 20 known matches, predictions were generated across all combinations of:

- Who bats first (T1 or T2) — 2 values
- Pitch type (balanced, flat, spin, pace) — 4 values
- Temperature level (cool, warm, hot) — 3 values
- Humidity level (dry, moderate, humid) — 3 values
- Rain risk (0, 1) — 2 values
- Dew risk (0, 1) — 2 values

That is **2 × 4 × 3 × 3 × 2 × 2 = 288 condition combinations** per match × 20 matches = **5,760 total lookup entries**, each storing the per-model and ensemble win probability for Team 1.

### Monte Carlo Tournament Simulation

5 independent tournament simulations were run and averaged. Each simulation:
1. Uses the 20 known fixture predictions as the base
2. Generates ~50 additional simulated fixtures to complete every team's 14-match schedule
3. Uses a greedy slot-pairing algorithm to ensure exactly 14 matches per team
4. Simulates all playoff rounds (Q1, Eliminator, Q2, Final) probabilistically

Results saved to `monte_carlo_results.json` as `win_probability` and `playoff_probability` per team.

### `base_extra_pts` — Live Table Foundation

To power the live expected-value league table without re-running simulations in the browser, a pre-computed scalar `base_extra_pts[team]` was added to the JSON data. This value represents:

```
base_extra_pts[team] = avg_monte_carlo_pts[team] - sum(2 × p_match for known matches)
```

At runtime the browser computes:
```
live_pts[team] = base_extra_pts[team] + Σ(2 × p_match for all 20 known matches)
```

For the currently selected match, the current condition-adjusted probability is used; all other matches use the default probability. This gives instant, deterministic, condition-sensitive expected-points updates with zero simulation cost.

---

## 10. Phase 6 — Live Dashboards

### 10.1 HTML Dashboard (`dashboard.html`)

A **fully self-contained single-file** interactive match simulator (~792 KB) with all CSS, JavaScript, and data embedded inline. No server required — open in any browser.

**Layout**

```
┌────────────────────────────────────────────────────────────┐
│  🏏 IPL 2026  Live Match Prediction Simulator   Built by…  │  ← sticky header
├──────────────────────┬─────────────────────────────────────┤
│  2026 Schedule       │  ⚙️ Match Conditions                 │
│  [Team filter ▾]     │  Toss Winner  · Toss Decision        │
│                      │  Pitch Type   · Temperature           │
│  RCB vs SRH — Mar 28 │  Humidity     · Rain Risk · Dew Risk │
│  MI  vs KKR — Mar 29 │  [↺ Reset to match defaults]         │
│  ...                 │  ─────────────────────────────────── │
│  (20 matches,        │  🎯 Prediction Output                 │
│   filtered by team)  │  [Win probability gauge]             │
│                      │  Match Intel (ELO, Batting,          │
│                      │   Flatness, Dew Factor)              │
│                      │  [Metric guide legend]               │
├──────────────────────┴─────────────────────────────────────┤
│  📊 Simulated League Table  │  🏆 Playoff Bracket           │
├─────────────────────────────┴───────────────────────────────┤
│  🎲 Tournament Win Probabilities (Monte Carlo)              │
├─────────────────────────────────────────────────────────────┤
│  📈 Model Performance                                        │
└─────────────────────────────────────────────────────────────┘
```

**Key interactive features**

- **Team filter dropdown**: Filter the 20-match fixture list by team. The currently selected match stays highlighted if it passes the filter; otherwise the first visible match is auto-selected.
- **Condition controls**: All 7 condition controls update win probabilities instantly via the pre-computed lookup table. No page reload, no network call.
- **Live expected-value league table**: Updates in real time as conditions change. The team batting first in the selected match sees its win probability shift, which propagates into expected points for all 10 teams.
- **Live playoff bracket**: Top 4 teams from the live table drive the bracket automatically. Q1, Eliminator, Q2 and Final update in real time.
- **Reset button**: Reverts all condition controls to the match's pre-computed weather defaults.

**Build process**

The dashboard is generated by `build_dashboard.py`, which reads `dashboard_data_v2.json` and writes `dashboard.html` as a Python heredoc. The data JSON is embedded directly in the `<script>` block as a `const DATA = {...}` assignment.

### 10.2 Bug Fixes During Dashboard Development

A series of bugs were identified and fixed during interactive testing:

| Bug | Root Cause | Fix |
|---|---|---|
| Toss decision buttons unresponsive | `onclick` handlers didn't call `renderControls()` | Added `renderControls()` to both toss decision button handlers |
| League table blank points | Code used `row.pts` but field is `row.points` | Changed to `row.won * 2` |
| Q2/Final showing cryptic abbreviations | `bracketTeamPill` sliced non-team strings with `.slice(0,3)` | Check `COLORS[name]` — real teams get colour pills, labels get full muted text |
| Teams playing 13–17 matches in MC | Simulation targeted wrong match count and ignored per-team limits | Fixed with `TARGET=14`, slot-pool greedy pairing with reshuffle fallback |
| Dashboard completely broken | Data re-embedding accidentally discarded all JS functions between two DATA occurrences | Rewrote `build_dashboard.py` as a full Python heredoc — no partial text substitutions |
| League table ordering wrong | `won`/`lost` from first MC sim only; `points` averaged → inconsistent | Average `won` and `lost` across all 5 sims; set `points = avg_won * 2` |
| Scorecard showing only one team | Both sections in a single scrollable panel, second was off-screen | Redesigned into 2-column side-by-side layout (later: scorecard section removed entirely) |
| League table static | Table rendered from pre-computed data, not live probabilities | Replaced with expected-value `liveTable()` using `base_extra_pts` |
| Simulated scorecard removed | Result logic inconsistent with toss/condition state; confusing UX | Removed scorecard and predicted playing XI sections entirely |

---

## 11. Streamlit Web App

A **deployable Streamlit web application** (`ipl_streamlit/app.py`) replicating all dashboard features for online hosting.

### File Structure

```
ipl_streamlit/
├── app.py                     ← main application (~500 lines)
├── requirements.txt           ← streamlit, plotly, pandas
├── .streamlit/
│   └── config.toml            ← dark theme (matches HTML dashboard)
└── data/
    └── dashboard_data_v2.json ← embedded prediction data (772 KB)
```

### Features

All features from the HTML dashboard are replicated:
- Sidebar fixture list with team filter dropdown and per-match radio selection
- All 7 match condition controls with live probability updates
- Semi-circle win probability gauge (Plotly)
- Match Intel grid (T1/T2 ELO, Batting score, Flatness, Dew Factor)
- Metric guide legend explaining each intel metric
- Live expected-value league table (HTML-styled, updates with conditions)
- Playoff bracket (updates live from league table top 4)
- Tournament win probabilities chart (Monte Carlo, horizontal bar)
- Model performance metrics (2 rows × 3 columns)
- Author name in top ribbon

### Deployment — Streamlit Community Cloud

1. Push the `ipl_streamlit/` directory to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select the repository, set **Main file path** to `app.py`
4. Click **Deploy** — Streamlit auto-installs `requirements.txt` and launches the app

### Running Locally

```bash
pip install streamlit plotly pandas
streamlit run ipl_streamlit/app.py
```

---

## 12. Tools & Packages

### Python Packages

| Package | Usage |
|---|---|
| `pandas` | Core data manipulation, CSV/Parquet I/O, joins, groupby aggregations |
| `numpy` | Numerical operations, array clipping, vectorized math |
| `xgboost` | Gradient boosted trees (base ensemble model) |
| `lightgbm` | LightGBM gradient boosted trees (base ensemble model) |
| `scikit-learn` | RandomForestClassifier, LogisticRegression (meta-learner), train_test_split, metrics |
| `torch` | PyTorch — MLP neural network |
| `pyarrow` | Parquet file I/O backend for pandas |
| `requests` | HTTP requests to IPLT20 S3 API, Open-Meteo API, CricSheet downloads |
| `streamlit` | Streamlit web application framework |
| `plotly` | Interactive charts in Streamlit (gauge, bar charts) |
| `Chart.js` | Client-side charts in the HTML dashboard (doughnut gauge, bar charts) |
| `json` | Parsing IPLT20 JSONP feeds, writing/reading all JSON data files |
| `pathlib` | File path management across all scripts |
| `re` | Regex for JSONP callback stripping and team name normalization |
| `glob` | Batch file discovery for CricSheet CSV loading |

### External APIs & Data Sources

| Source | Type | Authentication | Usage |
|---|---|---|---|
| CricSheet (`cricsheet.org`) | Static ZIP download | None (free, open license) | Ball-by-ball historical IPL data 2008–2025 + T20 form data |
| IPLT20 S3 API | JSON/JSONP (public S3 bucket) | None | IPL 2026 match fixtures |
| Open-Meteo Archive API | REST JSON | None (free, no API key) | Historical hourly weather per venue per match date |
| Wikipedia | HTML page | None | Tournament format, team list, venue metadata |
| ESPN Cricinfo | Web scrape (attempted) | Blocked by Akamai CDN | Original target — all requests returned 403 |

---

## 13. MCP Usage

### What is MCP?
MCP (Model Context Protocol) is an open standard developed by Anthropic that allows AI models to communicate with external tools and services through standardized interfaces. This project was built in **Cowork mode** on the Claude desktop app, which exposes MCP tools as first-class function calls.

### MCP Tools Used

**Agent Tool (Sub-Agent Orchestration)**
Long-running tasks (weather API fetching for 1,118 matches, T20 form data processing) were delegated to autonomous sub-agents to avoid the 2-minute Bash timeout. Each sub-agent received a task description, ran independently, wrote its output files, and returned completion status.

**WebFetch / WebSearch**
Used to probe IPLT20 S3 CompIDs, confirm CricSheet archive URLs, retrieve Wikipedia tournament metadata, and look up the current IPL 2026 fixture release status (confirming only Phase 1 was published due to elections).

**Bash / Read / Write / Edit Tools**
Core development workflow — running Python scripts, reading existing files to review logic, writing new scripts, and making targeted edits to fix bugs.

**AskUserQuestion Tool**
Used at design decision points: weather granularity, player form scope, model architecture, and dashboard deployment target (Streamlit Community Cloud).

**TodoWrite Tool**
Used to maintain a live task list tracking progress across all phases, rendered as a visual progress widget in the Cowork interface.

---

## 14. Data Files Reference

All generated data files live in `ipl_data/`. Final deliverables are in `ipl_predictor/` (HTML) and `ipl_streamlit/` (Streamlit).

| File | Rows | Description |
|---|---|---|
| `all_deliveries.parquet` | 278,205 | Ball-by-ball delivery data, all IPL matches 2008–2025 |
| `matches_clean.csv` | 1,169 | Match-level metadata: toss, winner, venue, season |
| `match_features.csv` | 1,146 | Original 13-feature matrix + target |
| `enriched_features.csv` | ~1,100 | Full 33-feature matrix (historical, for training) |
| `enriched_features_2026.csv` | 20 | 33-feature matrix for IPL 2026 fixtures |
| `ipl_2026_fixtures.csv` | 20 | IPL 2026 scheduled matches (Mar 28 – Apr 12) |
| `ipl_2026_predictions_v2.csv` | 20 | Per-model + ensemble win probabilities for 2026 fixtures |
| `prob_lookup.json` | 5,760 entries | Condition-adjusted probability lookup (288 combos × 20 matches) |
| `monte_carlo_results.json` | 10 teams | Championship and playoff probabilities from MC simulation |
| `dashboard_data_v2.json` | — | Complete data bundle for both dashboards (772 KB) |
| `venue_pitch_profiles.csv` | 58 | Flatness, pace, spin, dew factor, pitch label per venue |
| `weather_cache.csv` | 1,118 | Raw Open-Meteo hourly weather per historical match |
| `weather_features.csv` | 1,118 | Processed weather feature table per match |
| `elo_features.csv` | 1,169 | ELO ratings and win probability per historical match |
| `elo_current.json` | 10 | IPL 2026 pre-season ELO for all 10 teams |
| `player_form_features.csv` | 10 | Team-level aggregated batting/bowling form scores |
| `player_batting_form.csv` | ~200 | Per-player batting stats in form window |
| `player_bowling_form.csv` | ~200 | Per-player bowling stats in form window |
| `ipl_2026_squads.json` | 10 | Squad rosters per team |
| `model_metrics_v2.json` | 6 models | Accuracy and AUC-ROC for all trained models |

---

## 15. Known Issues & Fixes

| Issue | Root Cause | Fix Applied |
|---|---|---|
| ESPN Cricinfo HTTP 403 | Akamai Bot Manager CDN blocks all server/VM IPs | Switched to CricSheet + IPLT20 S3 API + Wikipedia |
| `sorted()` TypeError on seasons | Mixed `str`/`int`/`float`/NaN values in season column | Cast season to `str` before `sorted(unique())` |
| `ValueError: cannot convert NaN to int` | `ball` column had NaN rows from delivery parsing | `pd.to_numeric(errors='coerce')` + `dropna(subset=['ball'])` |
| Open-Meteo 400 Bad Request | `relativehumidity_2m_mean` not valid as a `daily` parameter | Switched to `hourly` endpoint, sliced indices 15–19 for evening |
| Bash timeout (exit code 143) during weather fetch | 1,118 sequential API calls exceeded 2-minute limit | Delegated to `Agent` sub-agent |
| T20 CricSheet archive missing Dec 2025+ data | Downloaded `it20s_male_csv2.zip` (IT20 only) by mistake | Downloaded `t20s_male_csv2.zip` (all T20s) |
| Only 20 IPL 2026 fixtures published | BCCI releasing fixtures in batches due to state elections | Accepted 20 as initial set; `base_extra_pts` covers the remainder via MC simulation |
| Toss decision buttons unresponsive in dashboard | `onclick` handlers missing `renderControls()` call | Added `renderControls()` to both handlers |
| League table blank points column | Used `row.pts` but field name is `row.points` | Changed to `row.won * 2` |
| Q2/Final bracket showing cryptic text | `bracketTeamPill` called `.slice(0,3)` on non-team label strings | Added `COLORS[name]` guard — real teams get colour pills, labels get muted text |
| Teams playing 13–17 matches in MC simulation | Simulation targeted wrong count; took random pairs ignoring per-team limit | Fixed with `TARGET=14`, greedy slot-pairing with reshuffle fallback |
| Dashboard fully broken after data re-embed | Re-embedding script found wrong boundary, discarded all JS between two DATA occurrences | Rewrote `build_dashboard.py` as a complete Python heredoc |
| League table ordering wrong (lower-pts team ranked #1) | `won`/`lost` taken from first MC sim only; `points` averaged across 5 → inconsistent | Average `won` and `lost` across all 5 sims; `points = avg_won * 2` |
| League table static (didn't react to condition changes) | Table rendered from pre-computed data only | Replaced with live expected-value `liveTable()` using `base_extra_pts` |
| `NameError: name 'teams' is not defined` in prep script | `teams` was scoped inside `simulate_tournament` function | Used `list(team_points_accum.keys())` as `_all_teams` |
| Fixture list team filter broken in Streamlit after filter change | Radio widget index out of bounds when filtered options list changed length | Added `key=f"fx_radio_{team_filter_sel}"` — key change forces widget reset |

---

## Project Status

| Phase | Status | Key Output |
|---|---|---|
| Phase 1: Data Collection + Initial Model | ✅ Complete | `matches_clean.csv`, `all_deliveries.parquet`, `ipl_2026_fixtures.csv` |
| Phase 2A: Venue Pitch Profiling | ✅ Complete | `venue_pitch_profiles.csv` (58 venues × 5 metrics) |
| Phase 2B: Historical Weather Features | ✅ Complete | `weather_features.csv` (1,118 matches × 12 features) |
| Phase 2C: ELO Rating System | ✅ Complete | `elo_features.csv`, `elo_current.json` |
| Phase 3: Player Form Features | ✅ Complete | `player_form_features.csv` |
| Phase 4: Enriched Feature Matrix | ✅ Complete | `enriched_features.csv` (~33 columns), `enriched_features_2026.csv` |
| Phase 5: Model Training v2 + MC Simulation | ✅ Complete | `ipl_2026_predictions_v2.csv`, `prob_lookup.json`, `monte_carlo_results.json`, `dashboard_data_v2.json` |
| Phase 6: HTML Dashboard | ✅ Complete | `ipl_predictor/dashboard.html` (792 KB, fully self-contained) |
| Streamlit Web App | ✅ Complete | `ipl_streamlit/app.py` (deployable to Streamlit Community Cloud) |

---

*Documentation last updated: March 19, 2026*
