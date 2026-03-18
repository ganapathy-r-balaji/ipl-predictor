<<<<<<< HEAD
# ipl-predictor
=======
# IPL 2026 Match Prediction System

**An end-to-end AI-powered cricket match prediction pipeline** using 17 years of historical IPL data (2008–2025), enriched with pitch profiles, weather data, ELO ratings, and player form features — culminating in an ensemble of ML models and a live interactive browser-based match simulator.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Phase 1 — Data Collection](#3-phase-1--data-collection)
4. [Phase 2A — Venue Pitch Profiling](#4-phase-2a--venue-pitch-profiling)
5. [Phase 2B — Historical Weather Features](#5-phase-2b--historical-weather-features)
6. [Phase 2C — ELO Rating System](#6-phase-2c--elo-rating-system)
7. [Phase 3 — Player Form Features](#7-phase-3--player-form-features)
8. [Phase 4 — Enriched Feature Matrix (Pending)](#8-phase-4--enriched-feature-matrix-pending)
9. [Phase 5 — Model Training (Pending)](#9-phase-5--model-training-pending)
10. [Phase 6 — Live Dashboard (Pending)](#10-phase-6--live-dashboard-pending)
11. [Tools & Packages](#11-tools--packages)
12. [MCP Usage](#12-mcp-usage)
13. [Data Files Reference](#13-data-files-reference)
14. [Known Issues & Fixes](#14-known-issues--fixes)

---

## 1. Project Overview

### Goal
Build a production-grade IPL match prediction system that:
- Ingests and processes 17 years of historical IPL ball-by-ball data (2008–2025)
- Fetches live IPL 2026 fixtures and team schedules
- Enriches each match with pitch conditions, weather, ELO ratings, and player form
- Trains an ensemble of ML models (XGBoost, LightGBM, Random Forest, PyTorch MLP)
- Runs 50,000 Monte Carlo simulations to estimate championship probabilities
- Delivers a fully client-side live match simulator dashboard with no backend server

### What Makes This Different From a Standard Kaggle Approach
- **Live data ingestion**: Fixtures sourced from the official IPLT20 S3 API, not a static dataset
- **Ball-by-ball granularity**: 278,205 individual deliveries processed to derive venue and team behavior
- **Temporal integrity**: Strict train/test time split (2008–2023 train, 2024–2025 test) to prevent data leakage
- **Contextual enrichment**: Every match gets its own pitch type, real historical weather, ELO-adjusted team strength, and recent player form
- **No backend required**: Pre-computed probability tables allow instant browser-side predictions

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
    └────────────────────────────────────────────────────────┘
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

---

## 3. Phase 1 — Data Collection

### What Was Done

#### 3.1 ESPN Cricinfo — Attempted & Blocked
The original plan was to scrape `espncricinfo.com/series/ipl-2026-1510719` directly for fixtures, team squads, venue data, and historical results. All requests — standard HTTP, `requests` with browser-like headers, Playwright headless browser automation, and direct API endpoint probing — returned **HTTP 403** errors.

**Root cause**: ESPN Cricinfo is protected by **Akamai Bot Manager CDN**, which fingerprints request patterns, TLS handshakes, and IP reputation. All cloud VM and server IP ranges are blocked by default.

**Why this matters**: The user's desired data source was completely inaccessible from a server environment, requiring a full pivot to alternative free and open data sources.

#### 3.2 Fallback Strategy — Three Alternative Sources

**Source 1: CricSheet (Primary historical data)**
- URL: `https://cricsheet.org/downloads/ipl_male_csv2.zip`
- What it provides: Ball-by-ball CSV data for every IPL match since 2008
- Format: One `{match_id}.csv` (deliveries) + one `{match_id}_info.csv` (metadata) per match
- Downloaded: **1,169 match files** covering 2008–2025 IPL seasons
- After loading and cleaning: **278,205 individual deliveries** in `all_deliveries.parquet`
- Match-level data: **1,169 rows × 21 columns** in `matches_clean.csv`

**Source 2: IPLT20 S3 API (IPL 2026 fixtures)**
- Base URL: `https://ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com/ipl/feeds/{CompID}-matchschedule.js`
- Discovery method: Scanned CompIDs 1–400 to identify the correct ID for IPL 2026 (found: **CompID 284**)
- Result: **20 fixtures** from March 28 – April 12, 2026 (remaining matches not yet published by IPL)
- Saved to: `ipl_2026_fixtures.csv`

**Source 3: Wikipedia (Tournament metadata)**
- Used as fallback for team names, venue locations, and format information
- Confirmed: 10 teams, double round-robin format, 74 league matches + 4 playoff matches

#### 3.3 Data Cleaning

Key normalizations applied:
- **Team name standardization**: 20+ historical team name variants (e.g., "Delhi Daredevils" → "DC", "Rising Pune Supergiant" → "RPS") mapped to canonical abbreviations
- **Season column**: Cast to `str` before sorting to avoid `TypeError` with mixed int/float NaN values
- **Ball column**: Parsed with `pd.to_numeric(errors='coerce')` and rows with NaN balls dropped to handle delivery numbering edge cases
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

Monte Carlo: 50,000 tournament simulations run, championship probabilities saved to `monte_carlo_results.json`.

---

## 4. Phase 2A — Venue Pitch Profiling

### Why This Was Added
Cricket outcomes are heavily influenced by pitch behavior. A flat batting pitch at Wankhede (Mumbai) produces totals 20–30 runs higher than a dry spin pitch at Chepauk (Chennai). Without encoding this, the model treats all venues identically — a significant blind spot that reduces predictive accuracy.

### What Was Done

All 278,205 deliveries were grouped by venue to compute four pitch metrics per ground.

#### 4.1 Flatness Index
```python
flatness = venue_match_avg_total / ipl_overall_avg_total
```
Measures how batter-friendly a pitch is relative to the IPL average. Values above 1.08 are labelled **"flat"** (high-scoring ground); below 0.92 is bowler-friendly.

#### 4.2 Pace Score
```python
pace_score = (pp_run_rate / ipl_pp_rr * 0.5) + (pace_wicket_pct * 0.5)
```
Combines powerplay run rate (proxy for seam movement and carry off the pitch) with the proportion of wickets falling to pace dismissal types (caught, bowled, hit-wicket). Higher scores indicate more assistance for fast bowling.

#### 4.3 Spin Score
```python
spin_score = (spin_wicket_pct * 0.6) + (max(0, 1 - mid_over_rr / death_rr) * 0.4)
```
Combines the proportion of spin-type dismissals (LBW, stumped) with middle-over slowdown (overs 7–15 versus death overs). Higher scores indicate more spin-friendly conditions.

#### 4.4 Dew Factor
```python
dew_factor = (2nd_inn_pp_rr / 2nd_inn_pp_balls) / (1st_inn_pp_rr / 1st_inn_pp_balls) - 1.0
```
Measures the 2nd innings powerplay run-rate premium over the 1st innings. A high dew factor means batting second is significantly easier because dew on the outfield makes the ball skid through and harder for bowlers to grip. This is critically important for evening matches at South/Southeast Asian grounds.

#### 4.5 Pitch Label Assignment
```python
if flatness > 1.08:      label = "flat"
elif spin_score > 0.18:  label = "spin"
elif pace_score > 1.05:  label = "pace"
else:                    label = "balanced"
```

**Results**: 58 unique IPL venues profiled. Notable findings:
- Mumbai/Wankhede: flatness ~1.15 → **flat**, high-scoring
- Chennai/Chepauk: spin_score ~0.22 → **spin**-dominant
- Delhi/Kotla: pace_score ~1.08 → **pace**-friendly, dry conditions

Saved to: `venue_pitch_profiles.csv` and `venue_pitch_profiles.json`

---

## 5. Phase 2B — Historical Weather Features

### Why This Was Added
Weather materially affects T20 cricket in several ways:
- **Humidity and dew** make the ball slippery for bowlers in the 2nd innings, strongly favouring the chasing team in evening matches
- **Rain** can cause Duckworth-Lewis adjustments or match interruptions, fundamentally altering the match
- **Temperature** correlates with ground moisture, atmospheric swing conditions, and player fatigue

Without weather data, the model cannot distinguish between a January morning match in Kolkata (cool, no dew) and an April evening match in Mumbai (hot, heavy dew) — two completely different match environments.

### What Was Done

#### 5.1 API Used — Open-Meteo Archive
- URL: `https://archive-api.open-meteo.com/v1/archive`
- **Free, requires no API key**
- Provides historical hourly weather for any latitude/longitude worldwide, going back to 1940
- Used the `hourly` endpoint (not `daily`) to extract **evening hours 15:00–19:00** (matching typical IPL evening match times in IST)

#### 5.2 Features Extracted Per Match

| Feature | Description |
|---|---|
| `temp_max` | Maximum temperature on match day (°C) |
| `temp_min` | Minimum temperature on match day (°C) |
| `temp_avg` | Average evening temperature (°C) |
| `precipitation` | Total rainfall on match day (mm) |
| `windspeed` | Average wind speed at 10m height (km/h) |
| `humidity_evening` | Average relative humidity 15:00–19:00 (%) |
| `cloudcover_evening` | Average cloud cover 15:00–19:00 (%) |
| `dewpoint_evening` | Average dewpoint temperature 15:00–19:00 (°C) |
| `is_hot` | Binary: temp_avg > 35°C |
| `rain_risk` | Binary: precipitation > 1mm |
| `dew_risk` | Binary: humidity_evening > 75% |
| `humidity_normalized` | humidity_evening / 100 (scaled 0–1 for ML) |

#### 5.3 Scale of Operation
- **1,118 historical matches** processed (all matches with valid date + venue coordinates)
- API calls batched by venue to minimize network overhead
- Long-running operation (~15 minutes total) handled via a sub-agent to avoid timeout
- Saved to: `weather_cache.csv` (raw) and `weather_features.csv` (processed 13-column feature table)

**Key fix**: The `relativehumidity_2m_mean` field does not exist as a `daily` parameter in Open-Meteo. Switched from `daily` endpoint to `hourly` and sliced time indices 15–19 manually for evening readings.

---

## 6. Phase 2C — ELO Rating System

### Why This Was Added
Standard win-rate features treat all historical wins equally. A win against the defending champion in a playoff should carry more weight than a win against a mid-table team in a dead rubber league match. ELO ratings provide a principled, mathematically grounded way to encode **relative team strength** with **automatic recency weighting** — stronger wins produce larger rating gains.

The ELO system was originally developed for chess rankings and is now used by FIDE, FIFA, the English Premier League analytics community, and academic sports prediction research.

### How ELO Works

Each team starts with a rating of 1500. After every match, ratings update as follows:

```python
INITIAL_ELO   = 1500
K_LEAGUE      = 32   # K-factor for league matches
K_PLAYOFF     = 48   # Higher K-factor for playoffs (bigger stakes = larger rating swings)

def expected_win(r1, r2):
    # Probability Team 1 beats Team 2 given current ratings
    return 1.0 / (1 + 10 ** ((r2 - r1) / 400))

def update_elo(r1, r2, result, k):
    # result = 1 if Team 1 won, 0 if Team 2 won
    e1 = expected_win(r1, r2)
    new_r1 = r1 + k * (result - e1)
    new_r2 = r2 + k * ((1 - result) - (1 - e1))
    return new_r1, new_r2
```

If a strong team (1600) beats a weak team (1400), this is the expected outcome and ratings barely move. But if the weak team upsets the strong team, a large rating swing occurs — the surprise result is appropriately reflected in future predictions.

### IPL 2026 Pre-Season ELO Ratings

| Team | ELO Rating | Context |
|---|---|---|
| RCB | 1616.2 | Defending IPL 2025 champions |
| GT | 1547.7 | Strong 2024 performance |
| PBKS | 1535.2 | Consistent recent form |
| KKR | 1534.2 | 2024 Champions |
| MI | 1526.8 | Multiple-time champions, stable base |
| DC | 1520.7 | Improved recent seasons |
| SRH | 1511.7 | 2024 runners-up |
| LSG | 1500.1 | Average historical performance |
| RR | 1475.9 | Below average recent form |
| CSK | 1468.5 | Recent decline, rebuilding roster |

Saved to: `elo_features.csv` (per-match, all 1,169 historical matches) and `elo_current.json` (current pre-season ratings).

---

## 7. Phase 3 — Player Form Features

### Why This Was Added
The original feature matrix treats both teams as static entities defined only by aggregate historical stats. But cricket is deeply individual — a team is only as strong as the players who are currently available and in form. A team missing its key strike bowler or with its top-order batter out of form is fundamentally weaker than the same team at full strength, and no aggregate historical stat captures that.

To encode **current player momentum**, we extract batting and bowling performance for every relevant player over the **3-month window immediately preceding IPL 2026**: December 28, 2025 – March 28, 2026.

### Data Sources for Form Window

**IPL 2025 Season Data**
- Source: CricSheet `ipl_male_csv2.zip` (already downloaded)
- 74 IPL 2025 matches covering domestic T20 performance for all IPL-contracted players

**T20 Internationals and Bilateral Series**
- Source: `https://cricsheet.org/downloads/t20s_male_csv2.zip`
- Full T20 CricSheet archive: 3,211 matches across all nations
- Filtered to Dec 28, 2025 – Mar 28, 2026: **158 matches** (T20 World Cup 2026 + bilateral series)
- Provides international T20 form for overseas IPL players (England, West Indies, South Africa, Australia, New Zealand)

**Total form window**: 232 matches, 53,340 deliveries.

### Batting Form Computation (Per Player)

```python
batting_score = (batting_avg * strike_rate / 100).clip(upper=500)
```

Where `batting_avg` = total runs / innings, `strike_rate` = (runs / balls faced) × 100, and `boundary_rate` = (4s + 6s) / balls faced. The composite score rewards both volume and aggression.

### Bowling Form Computation (Per Player)

```python
bowling_score = (wicket_rate * (9 / economy.clip(lower=1))).clip(upper=20)
```

Where `wicket_rate` = wickets / overs bowled, `economy` = runs / overs bowled. The composite score rewards both wicket-taking ability and economy.

### Team Aggregation

```python
# Batting unit: top 6 batters by batting_score
team_batting_score = top_6['batting_score'].mean()
team_batting_sr    = top_6['strike_rate'].mean()

# Bowling unit: top 4 bowlers by bowling_score
team_bowling_score = top_4['bowling_score'].mean()
team_bowling_econ  = top_4['economy'].mean()
```

Player-to-team mapping uses `ipl_2026_squads.json` (extracted from IPL 2025 info files as a proxy roster).

### Results Summary

| Team | Batting Score | Bowling Score | Bowling Economy |
|---|---|---|---|
| Rajasthan Royals | 54.96 (highest) | — | — |
| Royal Challengers Bangalore | — | 0.663 (highest) | — |
| Chennai Super Kings | — | — | 7.61 (best economy) |

Saved to: `player_form_features.csv`, `player_batting_form.csv`, `player_bowling_form.csv`, `ipl_2026_squads.json`

---

## 8. Phase 4 — Enriched Feature Matrix (Pending)

### Plan
Join all data sources into a single unified feature matrix with ~33 columns for model retraining.

| Feature Group | Features | Count |
|---|---|---|
| Original rolling stats | team form, H2H, venue win rates, toss, batting/bowling averages | 13 |
| ELO ratings | `t1_elo`, `t2_elo`, `elo_diff`, `elo_win_prob_t1` | 4 |
| Weather | `temp_avg`, `precipitation`, `humidity_evening`, `dewpoint_evening`, `rain_risk`, `dew_risk` | 6 |
| Pitch | `flatness_index`, `pace_score`, `spin_score`, `dew_factor` | 4 |
| Player form | `t1_batting_score`, `t1_bowling_score`, `t1_batting_sr`, `t1_bowling_econ` + same for t2 | 8 |

Join strategy: `match_id` for ELO and weather; `venue` name for pitch; team name for player form.

---

## 9. Phase 5 — Model Training (Pending)

### Planned Models

**PyTorch Neural Network (MLP)**
```
Architecture: Input(33) → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
                        → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
                        → Linear(64) → ReLU → Linear(1) → Sigmoid
```
Optimizer: Adam (lr=0.001), Loss: Binary Cross-Entropy, 50 epochs, batch size 64.

**Stacking Meta-Learner**
- Level 0 (base models): XGBoost + LightGBM + Random Forest + PyTorch MLP
- Level 1 (meta-learner): Logistic Regression trained on **out-of-fold predictions** from Level 0
- Out-of-fold training prevents data leakage from base models to the meta-learner

---

## 10. Phase 6 — Live Dashboard (Pending)

### Planned Architecture
A fully redesigned single-file `dashboard.html` running entirely in the browser with no backend server.

**Left panel**: Fixture selector — click any IPL 2026 match card

**Center panel**: Condition sliders:
- Temperature (°C)
- Humidity (%)
- Rain probability (%)
- Pitch type (flat / spin / pace / balanced)
- Dew factor (0–1)
- Toss winner + decision

**Right panel**:
- Animated probability gauge (win % for each team)
- Per-model breakdown bar chart (XGB / LGB / RF / MLP / Ensemble)
- Key factor contribution cards (what is driving the prediction)

**Bottom panel**:
- Tournament simulator with "Run 50k Sims" button
- Animated team bar chart showing championship probabilities

**Architecture decision**: All slider interactions trigger instant DOM updates using **pre-computed probability lookup tables** embedded as JSON in the HTML file — covering ~10,000 condition combinations per match. Zero latency, zero server calls, works completely offline.

---

## 11. Tools & Packages

### Python Packages

| Package | Usage |
|---|---|
| `pandas` | Core data manipulation, CSV/Parquet I/O, joins, groupby aggregations |
| `numpy` | Numerical operations, array clipping, vectorized math |
| `xgboost` | Gradient boosted trees (base ensemble model) |
| `lightgbm` | LightGBM gradient boosted trees (base ensemble model) |
| `scikit-learn` | RandomForestClassifier, LogisticRegression (meta-learner), train_test_split, accuracy_score |
| `torch` | PyTorch — MLP neural network (Phase 5, pending) |
| `pyarrow` | Parquet file I/O backend for pandas (`to_parquet`, `read_parquet`) |
| `requests` | HTTP requests to IPLT20 S3 API, Open-Meteo API, CricSheet downloads |
| `zipfile` | Extracting CricSheet `.zip` archives |
| `json` | Parsing IPLT20 JSONP feeds, writing ELO/squad/results JSON files |
| `pickle` | Serializing trained model objects to `models.pkl` |
| `datetime` | Date parsing, form window date filtering |
| `math` | ELO expected win probability formula |
| `re` | Regex for JSONP callback stripping and team name normalization |
| `glob` | Batch file discovery for CricSheet CSV loading |
| `os` / `pathlib` | File path management across all agents |
| `tqdm` | Progress bars for long-running loops (weather fetch, ELO computation) |
| `concurrent.futures` | Parallel weather API fetching (used in sub-agent) |

### External APIs & Data Sources

| Source | Type | Authentication | Usage |
|---|---|---|---|
| CricSheet (`cricsheet.org`) | Static ZIP download | None (free, open license) | Ball-by-ball historical IPL data 2008–2025 + T20 form data |
| IPLT20 S3 API | JSON/JSONP (public S3 bucket) | None | IPL 2026 match fixtures |
| Open-Meteo Archive API | REST JSON | None (free, no API key) | Historical hourly weather per venue per match date |
| Wikipedia | HTML page (requests + parsing) | None | Tournament format, team list, venue metadata |
| ESPN Cricinfo | Web scrape (attempted) | Blocked by Akamai CDN | Original target — all requests returned 403 |

### Development Environment

| Tool | Usage |
|---|---|
| Python 3.10 on Ubuntu 22.04 | All data processing and model training |
| Sandboxed Linux VM (Cowork) | Runtime environment |
| `pip` with `--break-system-packages` | Package installation in the VM environment |
| `bash` | Script execution, file management, package installation |

---

## 12. MCP Usage

### What is MCP?
MCP (Model Context Protocol) is an open standard developed by Anthropic that allows AI models like Claude to communicate with external tools, services, and data sources through standardized "connector" interfaces. In this project, Claude ran in **Cowork mode** — a desktop automation environment built on top of Claude and the Claude Agent SDK — which exposes MCP tools as first-class function calls alongside standard code execution.

### MCP Tools Used in This Project

#### Agent Tool (Sub-Agent Orchestration)
The most architecturally significant MCP usage was the **`Agent` tool** — Claude's ability to spawn autonomous sub-agents that independently handle long-running or isolated tasks without blocking the main conversation context.

**Weather Data Fetching (Phase 2B)**

Fetching weather for 1,118 historical matches requires 1,118+ sequential API calls to Open-Meteo, each with a ~0.5s network round-trip (~9 minutes total). Running this in the main context triggered a **2-minute timeout** (exit code 143). The solution:

```
Main Claude context:
└── Spawns Agent(subagent_type="general-purpose")
    ├── Receives: venue coordinates dict, match date list, output path
    ├── Executes: 1,118 Open-Meteo API calls sequentially
    ├── Writes: weather_cache.csv, weather_features.csv
    └── Returns: completion status to main context
```

The sub-agent ran to completion independently and returned control to the main conversation with the completed files. This is the key advantage of MCP-based sub-agent orchestration: long I/O-bound tasks can be delegated without any timeout risk to the primary context.

**T20 Form Data Processing (Phase 3)**

Downloading and extracting the 9.5MB T20 CricSheet archive (3,211 files), filtering to 158 form-window matches, and computing per-player statistics was also handled through the sub-agent pattern to avoid memory pressure and timeout constraints in the primary context.

#### WebFetch Tool
Used to retrieve content from URLs that are not executable APIs:
- Fetching CricSheet download pages to confirm correct zip archive URLs
- Probing IPLT20 S3 endpoints to discover valid CompIDs (1–400 scan)
- Fetching Wikipedia pages for tournament format and venue location data

**Limitation encountered**: WebFetch and all HTTP tools in this environment are subject to Akamai and CDN bot-detection systems. ESPN Cricinfo returned 403 for every method tried, including WebFetch with browser-like headers.

#### Bash Tool
Used for:
- `pip install` commands to add required packages
- `unzip` and file management operations
- Running Python scripts
- Verifying file sizes and output correctness

#### Read / Write / Edit Tools
Used extensively for:
- Reading existing agent Python files to review logic and catch bugs
- Writing new Python computation scripts for each phase
- Editing agent files to fix bugs (e.g., the `season` sorting TypeError)

#### TodoWrite Tool
Used to maintain a live task list tracking progress across phases. This renders as a visual progress widget in the Cowork interface, allowing the user to see exactly which phases are complete, in progress, and pending.

#### AskUserQuestion Tool
Used at critical design decision points where multiple valid approaches existed:

- **Weather granularity**: Full pitch report + granular weather vs simple temperature only
- **Player form scope**: All 11 players vs only star performers
- **Model scope**: Which additional model architectures to add beyond the tree ensemble
- **Dashboard architecture**: Server-side dynamic computation vs client-side pre-computed tables

This is an important MCP interaction pattern — rather than making arbitrary design decisions unilaterally, the tool surfaces explicit choices to the user and locks in requirements before building, preventing wasted effort on the wrong direction.

### MCP Tools Available But Not Used

| MCP Tool | Reason Not Used |
|---|---|
| Claude in Chrome (browser automation) | Could theoretically automate Cricinfo in a real browser window, but Akamai also blocks headless/automated browsers by fingerprinting automation signals (CDP protocol detection, JS environment leakage). Not attempted. |
| Google Drive / Sheets connectors | Not needed — all data stored locally in the sandboxed VM filesystem |
| Slack / email connectors | Not applicable to this data science project |

---

## 13. Data Files Reference

All generated data files are in the working data directory (`/ipl_data/` in the session VM). Final agent code is in `agents/`.

| File | Rows | Description |
|---|---|---|
| `all_deliveries.parquet` | 278,205 | Ball-by-ball delivery data, all IPL matches 2008–2025 |
| `matches_clean.csv` | 1,169 | Match-level metadata: toss, winner, venue, season |
| `match_features.csv` | 1,146 | Original 13-feature matrix + target |
| `ipl_2026_fixtures.csv` | 20 | IPL 2026 scheduled matches (Mar 28–Apr 12) |
| `ipl_2026_predictions.csv` | 20 | Win probability per 2026 fixture (Phase 1 model) |
| `monte_carlo_results.json` | — | Championship probabilities from 50,000 simulations |
| `models.pkl` | — | Serialized XGB + LGB + RF ensemble (Phase 1 model) |
| `venue_pitch_profiles.csv` | 58 | Flatness, pace, spin, dew factor, pitch label per venue |
| `venue_pitch_profiles.json` | 58 | Same as above in JSON format |
| `weather_cache.csv` | 1,118 | Raw Open-Meteo hourly weather per historical match |
| `weather_features.csv` | 1,118 | Processed 12-column weather feature table per match |
| `elo_features.csv` | 1,169 | ELO ratings and win probability per historical match |
| `elo_current.json` | 10 | IPL 2026 pre-season ELO for all 10 teams |
| `player_batting_form.csv` | ~200 | Per-player batting stats in form window |
| `player_bowling_form.csv` | ~200 | Per-player bowling stats in form window |
| `player_form_features.csv` | 10 | Team-level aggregated batting/bowling form scores |
| `ipl_2026_squads.json` | 10 | Squad rosters per team (IPL 2025 proxy) |

---

## 14. Known Issues & Fixes

| Issue | Root Cause | Fix Applied |
|---|---|---|
| ESPN Cricinfo HTTP 403 | Akamai Bot Manager CDN blocks all server/VM IP ranges | Switched to CricSheet + IPLT20 S3 API + Wikipedia |
| `sorted()` TypeError on seasons | Mixed `str`/`int`/`float`/NaN values in season column after CSV load | Cast season to `str` before calling `sorted(unique())` |
| `ValueError: cannot convert NaN to int` | `ball` column had NaN rows from delivery parsing edge cases | `pd.to_numeric(errors='coerce')` + `dropna(subset=['ball'])` |
| `ImportError: Missing optional dependency 'pyarrow'` | `pyarrow` not pre-installed in VM | `pip install pyarrow --break-system-packages` |
| Open-Meteo 400 Bad Request | `relativehumidity_2m_mean` is not a valid `daily` parameter | Switched to `hourly` endpoint, extracted indices 15–19 for evening |
| Timeout (exit code 143) during weather fetch | 1,118 sequential API calls exceeded 2-minute Bash timeout | Delegated to `Agent` sub-agent which ran the full loop independently |
| T20 CricSheet archive missing Dec 2025+ data | Downloaded `it20s_male_csv2.zip` (IT20 only, ends 2024) by mistake | Downloaded `t20s_male_csv2.zip` (all T20s) which contained 158 Dec 2025+ matches |
| Only 20 IPL 2026 fixtures published | Tournament starts Mar 28, 2026; IPL publishes fixture batches progressively | Accepted 20 fixtures as initial set; rest to be fetched progressively via IPLT20 S3 API |

---

## Project Status

| Phase | Status | Key Output |
|---|---|---|
| Phase 1: Data Collection + Initial Model | ✅ Complete | `matches_clean.csv`, `all_deliveries.parquet`, `ipl_2026_fixtures.csv`, initial `models.pkl` |
| Phase 2A: Venue Pitch Profiling | ✅ Complete | `venue_pitch_profiles.csv` (58 venues × 5 metrics) |
| Phase 2B: Historical Weather Features | ✅ Complete | `weather_features.csv` (1,118 matches × 12 features) |
| Phase 2C: ELO Rating System | ✅ Complete | `elo_features.csv`, `elo_current.json` |
| Phase 3: Player Form Features | ✅ Complete | `player_form_features.csv`, `player_batting_form.csv`, `player_bowling_form.csv` |
| Phase 4: Enriched Feature Matrix | 🔜 Next session | `enriched_features.csv` (~33 columns) |
| Phase 5: PyTorch MLP + Stacking | 🔜 Pending | `models_v2.pkl` |
| Phase 6: Live Match Simulator Dashboard | 🔜 Pending | `dashboard.html` (fully client-side) |

---

*Documentation last updated: March 16, 2026*
>>>>>>> dfef71a (Initial commit: IPL 2026 HTML prediction dashboard)
