# IPL 2026 Prediction System — All Prompts Log

This file records every user prompt that drove the design and development of the IPL 2026 Match Prediction System. Prompts are listed in chronological order, annotated with the phase they triggered.

---

## Prompt 1 — Project Kickoff & Data Source

> "no. this is not going to be a kaggle dataset. Here is what I want to do. I want to scrape schedules data from https://www.espncricinfo.com/series/ipl-2026-1510719/match-schedule-fixtures-and-results. The IPL website from Cricinfo is here (https://www.espncricinfo.com/series/ipl-2026-1510719). This website has information about IPL fixtures, results, playgrounds, squads from the previous years."

**What this triggered:** Abandoned the Kaggle dataset approach. Pivoted to scraping live IPL 2026 fixtures, historical match data, player squads, and venue data directly from ESPN Cricinfo. This led to the discovery of Akamai CDN blocking (403 errors on all VM/server IPs), which forced the fallback strategy using CricSheet + IPLT20 S3 API.

---

## Prompt 2 — Feature Expansion & Model Upgrade Request

> "for each match that was played in the past and matches to be played this year, let's take into account the weather conditions, soil data. For the current season, lets take into account the players' performances over the last 3 months and use it to create a feature that helps with their future performances. Also, I don't want to just stop at XGBoost, LightGBM, Random Forest and Ensemble method. Also, hate the dashboard layout. I want it to be more interactive. Let's plan step by step."

**What this triggered:** A complete V2 redesign of the entire pipeline. Four major additions were planned:
1. **Pitch/Soil profiling** per venue (bounce, pace, spin, dew factor)
2. **Weather features** per historical match (temperature, humidity, rain, dew)
3. **Player form features** from the last 3 months before the season
4. **Expanded ML models** beyond XGB/LGB/RF
5. **New interactive dashboard** replacing the static one

---

## Prompt 3 — Clarifying Question Answers (Weather & Player Features)

> [Answered AskUserQuestion]:
> - Weather granularity = **Full pitch report** (bounce, pace, spin, dew factor) **+ granular weather** (temperature, humidity, rain risk, dew risk)
> - Player window = **Full squad, all 11 players**, weighted by **recent form**
> - Models = **Neural Network (TabNet/MLP) + Bayesian ELO** in addition to tree-based ensemble
> - Dashboard = **Live match simulator** with adjustable sliders for conditions

**What this triggered:** Locked the scope for Phase 2 (pitch + weather + ELO) and Phase 3 (player form).

---

## Prompt 4 — Clarifying Question Answers (Dashboard & Time Window)

> [Answered AskUserQuestion]:
> - Dashboard updates = **Instantly in-browser** using **pre-computed probability tables** (no backend server)
> - Player form window = **3 months prior to the beginning of the season** (Dec 28, 2025 – Mar 28, 2026)

**What this triggered:** Confirmed the client-side-only dashboard architecture (no Flask/FastAPI server required) and defined the exact date range for player form extraction.

---

## Prompt 5 — Phase Progression

> "Lets proceed to the next phase."

**What this triggered:** Moved from Phase 2 (pitch + weather + ELO features) to Phase 3 (player form features). This single prompt was used multiple times to advance through each completed phase.

---

## Prompt 6 — Documentation Request

> "save all the prompts to a md file. Also, create another readme file explaining all the phases step by step and explain in detail what you have done in each step, and why you have done that. Also list all the tools and packages used. Explain if you have used MCP, and how you have done it."

**What this triggered:** Creation of this `PROMPTS.md` file and the detailed `README.md` documentation file covering the full project.

---

## Prompt 7 — Session Close

> "I will see you tomorrow."

**What this triggered:** Session paused after completing Phase 3. Next session picks up at Phase 4 (enriched feature matrix rebuild).

---

*Last updated: March 16, 2026*
