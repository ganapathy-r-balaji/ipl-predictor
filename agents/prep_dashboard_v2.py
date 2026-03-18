"""
Prepare enhanced dashboard data (v2) with:
  - Player rosters + roles per team
  - Simulated scorecards per match
  - Time-based weather defaults
  - Tournament simulation (points table + playoff bracket)
"""

import json, random, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

random.seed(42)
np.random.seed(42)

DATA_DIR = Path('/sessions/peaceful-cool-keller/ipl_data')

# ── load data ─────────────────────────────────────────────
squads_raw  = json.load(open(DATA_DIR / 'ipl_2026_squads.json'))
batting_df  = pd.read_csv(DATA_DIR / 'player_batting_form.csv')
bowling_df  = pd.read_csv(DATA_DIR / 'player_bowling_form.csv')
fixtures    = pd.read_csv(DATA_DIR / 'ipl_2026_fixtures.csv')
preds       = pd.read_csv(DATA_DIR / 'ipl_2026_predictions_v2.csv')
lookup      = json.load(open(DATA_DIR / 'prob_lookup.json'))
mc          = json.load(open(DATA_DIR / 'monte_carlo_results.json'))
elo_cur     = json.load(open(DATA_DIR / 'elo_current.json'))
metrics     = json.load(open(DATA_DIR / 'model_metrics_v2.json'))
enrich      = pd.read_csv(DATA_DIR / 'enriched_features_2026.csv')

batting_df['player'] = batting_df['player'].str.strip()
bowling_df['player'] = bowling_df['player'].str.strip()
bat_idx  = batting_df.set_index('player')
bowl_idx = bowling_df.set_index('player')

# ── city weather profiles (base values per city) ─────────
CITY_WEATHER = {
    # city: {afternoon: {temp,hum,dew_risk}, evening: {temp,hum,dew_risk}}
    'Bengaluru':      {'afternoon': (32,55,False), 'evening': (27,68,True)},
    'Mumbai':         {'afternoon': (34,60,True),  'evening': (29,75,True)},
    'Chennai':        {'afternoon': (36,65,True),  'evening': (31,78,True)},
    'Kolkata':        {'afternoon': (34,58,False),  'evening': (28,72,True)},
    'Hyderabad':      {'afternoon': (36,50,False), 'evening': (30,65,True)},
    'Delhi':          {'afternoon': (34,45,False), 'evening': (27,60,False)},
    'Ahmedabad':      {'afternoon': (36,40,False), 'evening': (30,55,False)},
    'Jaipur':         {'afternoon': (36,38,False), 'evening': (28,52,False)},
    'Lucknow':        {'afternoon': (34,48,False), 'evening': (27,63,True)},
    'New Chandigarh': {'afternoon': (30,42,False), 'evening': (23,55,False)},
    'Mullanpur':      {'afternoon': (30,42,False), 'evening': (23,55,False)},
    'Guwahati':       {'afternoon': (30,70,True),  'evening': (25,80,True)},
    'Dharamsala':     {'afternoon': (22,55,False), 'evening': (16,65,False)},
    'Ranchi':         {'afternoon': (33,52,False), 'evening': (27,67,True)},
    'Cuttack':        {'afternoon': (34,62,True),  'evening': (29,74,True)},
}

def get_weather_defaults(city, time_ist):
    profile = CITY_WEATHER.get(city, {'afternoon': (33,50,False), 'evening': (28,65,True)})
    slot = 'afternoon' if time_ist.startswith('15') or time_ist.startswith('14') else 'evening'
    temp, hum, dew = profile[slot]
    # Map to slider levels
    temp_level = 'cool' if temp < 26 else ('hot' if temp >= 34 else 'warm')
    hum_level  = 'dry' if hum < 55 else ('humid' if hum >= 70 else 'moderate')
    return {'temp_level': temp_level, 'hum_level': hum_level,
            'dew_risk': int(dew), 'rain_risk': 0, 'temp_c': temp, 'humidity_pct': hum}

# ── known wicket-keepers ───────────────────────────────────
WK_PLAYERS = {
    'MS Dhoni', 'Ishan Kishan', 'KL Rahul', 'Dinesh Karthik', 'Sanju Samson',
    'Quinton de Kock', 'Phil Salt', 'Jos Buttler', 'Heinrich Klaasen',
    'Nicholas Pooran', 'Rishabh Pant', 'Abishek Porel', 'Dhruv Jurel',
    'Jitesh Sharma', 'Prabhsimran Singh', 'Matthew Wade', 'Tim Seifert',
    'Josh Inglis', 'Sam Billings', 'Tanmay Bhatia',
}

# Common opening batters
OPENERS = {
    'Rohit Sharma', 'Shubman Gill', 'Virat Kohli', 'Faf du Plessis', 'F du Plessis',
    'KL Rahul', 'Quinton de Kock', 'Phil Salt', 'Travis Head', 'T Head',
    'David Warner', 'Prabhsimran Singh', 'Yashasvi Jaiswal', 'YS Jaiswal',
    'Abhishek Sharma', 'Jos Buttler', 'J Buttler', 'Jake Fraser-McGurk',
    'J Fraser-McGurk', 'T Stubbs', 'Ishan Kishan', 'Ruturaj Gaikwad', 'RG Gaikwad',
    'Wriddhiman Saha', 'Devon Conway', 'DP Conway', 'Sai Sudharsan',
}

# ── build player profiles per team ───────────────────────
def build_team_roster(team_name):
    players = squads_raw.get(team_name, [])
    roster  = []

    for p in players:
        has_bat  = p in bat_idx.index
        has_bowl = p in bowl_idx.index

        bat_stats  = bat_idx.loc[p].to_dict() if has_bat  else {}
        bowl_stats = bowl_idx.loc[p].to_dict() if has_bowl else {}

        bat_score  = float(bat_stats.get('batting_score', 0))
        bowl_score = float(bowl_stats.get('bowling_score', 0))
        bat_avg    = float(bat_stats.get('batting_avg', 15))
        bat_sr     = float(bat_stats.get('strike_rate', 120))
        bowl_econ  = float(bowl_stats.get('economy', 9))
        wicket_r   = float(bowl_stats.get('wicket_rate', 0))
        innings    = int(bat_stats.get('innings', 0))
        overs      = float(bowl_stats.get('overs_bowled', 0))

        # Assign role
        is_wk = p in WK_PLAYERS
        if bat_score > 20 and bowl_score > 0.3:
            role = 'All-Rounder'
        elif bat_score > 20 or (bat_avg > 20 and innings >= 3):
            role = 'Wicket-Keeper' if is_wk else 'Batsman'
        elif bowl_score > 0.2 or overs >= 4:
            role = 'Bowler'
        else:
            role = 'Wicket-Keeper' if is_wk else ('Batsman' if bat_score > 0 else 'Bowler')

        is_opener = p in OPENERS

        roster.append({
            'name':       p,
            'role':       role,
            'is_wk':      is_wk,
            'is_opener':  is_opener,
            'bat_score':  round(bat_score, 2),
            'bowl_score': round(bowl_score, 3),
            'bat_avg':    round(max(bat_avg, 5), 1),
            'bat_sr':     round(max(bat_sr, 80), 1),
            'bowl_econ':  round(bowl_econ, 2),
            'wicket_rate': round(wicket_r, 3),
            'innings':    innings,
            'overs':      round(overs, 1),
        })

    return roster

# ── pick playing 11 ───────────────────────────────────────
def pick_playing_11(roster):
    # Sort candidates
    batters     = sorted([p for p in roster if p['role'] in ('Batsman','Wicket-Keeper')],
                          key=lambda x: -x['bat_score'])
    bowlers     = sorted([p for p in roster if p['role'] == 'Bowler'],
                          key=lambda x: -x['bowl_score'])
    allrounders = sorted([p for p in roster if p['role'] == 'All-Rounder'],
                          key=lambda x: -(x['bat_score'] + x['bowl_score'] * 30))
    wks         = [p for p in roster if p['is_wk']]

    selected = []
    names_selected = set()

    def add(p):
        if p['name'] not in names_selected and len(selected) < 11:
            selected.append(p)
            names_selected.add(p['name'])

    # 1 WK
    for wk in sorted(wks, key=lambda x: -x['bat_score']):
        add(wk); break

    # 2 all-rounders
    for ar in allrounders[:3]:
        add(ar)

    # 5 pure batters
    for b in batters[:6]:
        add(b)

    # 4 pure bowlers
    for bw in bowlers[:5]:
        add(bw)

    # Pad to 11 from remaining
    remaining = [p for p in roster if p['name'] not in names_selected]
    remaining_sorted = sorted(remaining, key=lambda x: -(x['bat_score'] + x['bowl_score']))
    for p in remaining_sorted:
        add(p)

    playing_11 = selected[:11]

    # Order by batting position
    openers = [p for p in playing_11 if p['is_opener']]
    top_order = [p for p in playing_11 if not p['is_opener'] and p['role'] in ('Batsman','Wicket-Keeper')]
    ar_list  = [p for p in playing_11 if p['role'] == 'All-Rounder']
    lower    = [p for p in playing_11 if p['role'] == 'Bowler']

    ordered = openers + [p for p in top_order if p not in openers] + ar_list + lower
    # Fill any gaps
    seen = set(p['name'] for p in ordered)
    for p in playing_11:
        if p['name'] not in seen:
            ordered.append(p)

    return ordered[:11]

def pick_impact_sub(roster, playing_11):
    """Pick best unused player as impact sub."""
    p11_names = {p['name'] for p in playing_11}
    bench = [p for p in roster if p['name'] not in p11_names]
    if not bench:
        return None
    # Prefer all-rounders, then best batsman or bowler
    bench_sorted = sorted(bench,
        key=lambda x: -(x['bat_score'] * 0.6 + x['bowl_score'] * 30 * 0.4))
    sub = bench_sorted[0]
    # Reason
    if sub['role'] == 'All-Rounder':
        reason = f"Extra batting depth + {sub['wicket_rate']:.2f} wkts/over bowling option"
    elif sub['role'] == 'Bowler':
        reason = f"Specialist bowling (econ {sub['bowl_econ']:.1f}, {sub['wicket_rate']:.2f} wkts/over)"
    else:
        reason = f"Batting firepower (avg {sub['bat_avg']:.0f}, SR {sub['bat_sr']:.0f})"
    return {'player': sub, 'reason': reason}

# ── simulate batting scorecard ────────────────────────────
def simulate_scorecard(batting_team_11, bowling_team_11, target=None, venue_fl=1.0):
    """
    Simulate a T20 innings given batting and bowling sides.
    Returns dict with runs/balls/4s/6s per batsman, and bowling figures.
    """
    # Estimate total based on batting quality vs bowling quality
    avg_bat_score = np.mean([p['bat_score'] for p in batting_team_11[:7]])
    avg_bowl_score = np.mean([p['bowl_score'] for p in bowling_team_11 if p['bowl_score'] > 0][:5] or [0.3])

    # Base total: ~155 average IPL score
    base_total = 155 + (avg_bat_score - 30) * 1.2 - (avg_bowl_score - 0.35) * 40
    base_total *= venue_fl  # pitch flatness adjustment
    if target:
        # Chasing — slightly harder, pressure factor
        chase_adj = 0.88 + random.gauss(0, 0.07)
        base_total = target * chase_adj

    total_runs   = int(np.clip(random.gauss(base_total, 18), 90, 240))
    total_wickets = int(np.clip(random.gauss(6.5, 2.0), 0, 10))
    total_balls  = 120

    # Distribute runs across batting positions
    # Openers get more balls, lower order less
    position_weights = [0.22, 0.18, 0.16, 0.13, 0.10, 0.08, 0.06, 0.04, 0.01, 0.01, 0.01]
    batting_entries = []
    runs_left = total_balls
    distributed_runs = 0

    for i, batter in enumerate(batting_team_11):
        weight = position_weights[i] if i < len(position_weights) else 0.005
        balls_b = max(1, int(round(total_balls * weight * random.gauss(1.0, 0.15))))
        # SR based on position and player's historical SR
        sr_adj = batter['bat_sr'] * random.gauss(1.0, 0.12)
        sr_adj = max(60, min(220, sr_adj))
        runs_b = int(balls_b * sr_adj / 100)
        # 4s and 6s
        boundary_pct = max(0, random.gauss(0.22, 0.05))
        fours  = int(balls_b * boundary_pct * 0.65)
        sixes  = int(balls_b * boundary_pct * 0.35)
        is_out = (i < total_wickets)
        distributed_runs += runs_b
        batting_entries.append({
            'name':   batter['name'],
            'role':   batter['role'],
            'runs':   runs_b,
            'balls':  balls_b,
            'fours':  fours,
            'sixes':  sixes,
            'sr':     round(runs_b / max(balls_b, 1) * 100, 1),
            'out':    is_out,
        })

    # Scale to actual total
    scale = total_runs / max(distributed_runs, 1)
    for e in batting_entries:
        e['runs']  = max(0, int(e['runs'] * scale))
        e['fours'] = min(e['runs']//4, e['fours'])
        e['sixes'] = min(e['runs']//6, e['sixes'])

    # Bowling figures
    bowlers_pool = sorted([p for p in bowling_team_11 if p['bowl_score'] > 0 or p['overs'] > 0],
                           key=lambda x: -x['bowl_score'])[:6]
    if len(bowlers_pool) < 4:
        bowlers_pool = [p for p in bowling_team_11][:6]

    bowling_entries = []
    wickets_left = total_wickets
    over_shares = distribute_overs(len(bowlers_pool), 20)

    for i, bowler in enumerate(bowlers_pool):
        overs_b = over_shares[i]
        balls_b = int(overs_b * 6)
        econ_adj = bowler['bowl_econ'] * random.gauss(1.0, 0.12)
        econ_adj = max(4, min(16, econ_adj))
        runs_b   = int(overs_b * econ_adj)
        wkts_share = min(wickets_left, max(0, round(bowler['wicket_rate'] * overs_b * random.gauss(1.0, 0.3))))
        wkts_share = min(wkts_share, 4)
        wickets_left = max(0, wickets_left - wkts_share)
        maiden = max(0, int(overs_b * max(0, (7 - econ_adj) / 10)))
        bowling_entries.append({
            'name':    bowler['name'],
            'overs':   overs_b,
            'maidens': maiden,
            'runs':    runs_b,
            'wickets': wkts_share,
            'economy': round(runs_b / max(overs_b, 0.1), 2),
        })

    return {
        'total':    total_runs,
        'wickets':  total_wickets,
        'batting':  batting_entries,
        'bowling':  bowling_entries,
    }

def distribute_overs(n_bowlers, total=20):
    """Distribute 20 overs across n bowlers (max 4 each)."""
    shares = []
    per = total // n_bowlers
    rem = total - per * n_bowlers
    for i in range(n_bowlers):
        o = min(4, per + (1 if i < rem else 0))
        shares.append(o)
    return shares

# ── tournament simulation (points table) ─────────────────
def simulate_tournament(fixture_df, pred_df):
    """
    Simulate full 74-match double round robin + playoffs.
    Returns points_table, playoff_bracket.
    Since only 20 matches are published, we repeat the 20-fixture
    win-probability pattern for the remaining 54 (3 full rounds + partial).
    """
    teams = [
        'Royal Challengers Bengaluru', 'Chennai Super Kings', 'Mumbai Indians',
        'Kolkata Knight Riders', 'Sunrisers Hyderabad', 'Delhi Capitals',
        'Rajasthan Royals', 'Punjab Kings', 'Gujarat Titans', 'Lucknow Super Giants',
    ]
    points = {t: 0 for t in teams}
    wins   = {t: 0 for t in teams}
    losses = {t: 0 for t in teams}
    played = {t: 0 for t in teams}
    nrr    = {t: 0.0 for t in teams}

    pred_map = dict(zip(pred_df['match_id'].astype(str),
                        pred_df['p_t1_avg'].values))

    # All 20 published fixtures
    for _, fx in fixture_df.iterrows():
        mid = str(fx['match_id'])
        t1 = fx['team1']; t2 = fx['team2']
        p_t1 = float(pred_map.get(mid, 0.5))
        # Simulate: use probability as win chance
        t1_wins = random.random() < p_t1
        if t1_wins:
            points[t1] += 2; wins[t1]  += 1; losses[t2] += 1
            nrr[t1] += random.gauss(0.35, 0.2)
            nrr[t2] -= random.gauss(0.35, 0.2)
        else:
            points[t2] += 2; wins[t2]  += 1; losses[t1] += 1
            nrr[t2] += random.gauss(0.35, 0.2)
            nrr[t1] -= random.gauss(0.35, 0.2)
        played[t1] += 1; played[t2] += 1

    # Simulate remaining ~54 matches using ELO probs
    from itertools import combinations
    all_pairs = list(combinations(teams, 2))
    # Each pair plays 2x home+away in full DRR; we've covered 20 so extend
    extra_fixtures = []
    for t1, t2 in all_pairs:
        extra_fixtures.extend([(t1, t2), (t2, t1)])
    random.shuffle(extra_fixtures)
    remaining_needed = 74 - 20  # 54 matches
    for t1, t2 in extra_fixtures[:remaining_needed]:
        r1 = elo_cur.get(t1, 1500); r2 = elo_cur.get(t2, 1500)
        p_t1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        t1_wins = random.random() < p_t1
        if t1_wins:
            points[t1] += 2; wins[t1]  += 1; losses[t2] += 1
            nrr[t1] += random.gauss(0.25, 0.2)
            nrr[t2] -= random.gauss(0.25, 0.2)
        else:
            points[t2] += 2; wins[t2]  += 1; losses[t1] += 1
            nrr[t2] += random.gauss(0.25, 0.2)
            nrr[t1] -= random.gauss(0.25, 0.2)
        played[t1] += 1; played[t2] += 1

    # Build sorted table
    table = sorted(teams,
        key=lambda t: (-points[t], -nrr[t]))

    table_rows = []
    for i, t in enumerate(table):
        table_rows.append({
            'pos': i+1, 'team': t,
            'played': played[t], 'won': wins[t],
            'lost': losses[t], 'points': points[t],
            'nrr': round(nrr[t], 3),
            'qualified': i < 4,
        })

    # Playoff bracket
    top4 = table[:4]
    q1   = {'match': 'Qualifier 1', 'team1': top4[0], 'team2': top4[1],
            'winner_goes': 'Final', 'loser_goes': 'Qualifier 2',
            'note': '1st vs 2nd — winner goes direct to Final'}
    elim = {'match': 'Eliminator', 'team1': top4[2], 'team2': top4[3],
            'winner_goes': 'Qualifier 2', 'loser_goes': 'Eliminated',
            'note': '3rd vs 4th — loser is eliminated'}
    q2   = {'match': 'Qualifier 2', 'team1': f'Loser of Q1', 'team2': f'Winner of Elim',
            'winner_goes': 'Final', 'loser_goes': 'Eliminated',
            'note': 'Last chance to reach the Final'}
    final= {'match': 'Final', 'team1': f'Winner of Q1', 'team2': f'Winner of Q2',
            'note': 'IPL 2026 Champion crowned'}

    return table_rows, [q1, elim, q2, final]

# Run 5 simulations and average for stability
N_SIMS = 5
all_tables = []
all_brackets = []
for _ in range(N_SIMS):
    t, b = simulate_tournament(fixtures, preds)
    all_tables.append(t)
    all_brackets.append(b)

# Average points/NRR
from collections import Counter
team_points_accum = defaultdict(list)
team_nrr_accum    = defaultdict(list)
for table in all_tables:
    for row in table:
        team_points_accum[row['team']].append(row['points'])
        team_nrr_accum[row['team']].append(row['nrr'])

avg_points = {t: np.mean(v) for t, v in team_points_accum.items()}
avg_nrr    = {t: np.mean(v) for t, v in team_nrr_accum.items()}

final_table = sorted(avg_points.keys(), key=lambda t: (-avg_points[t], -avg_nrr[t]))
final_table_rows = []
for i, t in enumerate(final_table):
    sample = next(r for r in all_tables[0] if r['team'] == t)
    final_table_rows.append({
        'pos': i+1, 'team': t,
        'played': sample['played'],
        'won': sample['won'],
        'lost': sample['lost'],
        'points': round(avg_points[t], 1),
        'nrr': round(avg_nrr[t], 3),
        'qualified': i < 4,
    })

top4_teams = [r['team'] for r in final_table_rows[:4]]
bracket = [
    {'match': 'Qualifier 1', 'team1': top4_teams[0], 'team2': top4_teams[1],
     'note': '1st vs 2nd — winner goes direct to Final'},
    {'match': 'Eliminator',  'team1': top4_teams[2], 'team2': top4_teams[3],
     'note': '3rd vs 4th — loser eliminated'},
    {'match': 'Qualifier 2', 'team1': f'Loser of Q1 ({top4_teams[1][:3]})',
     'team2': f'Elim Winner ({top4_teams[2][:3]})',
     'note': 'Last chance match — winner reaches Final'},
    {'match': 'Final', 'team1': f'Q1 Winner ({top4_teams[0][:3]})',
     'team2': f'Q2 Winner', 'note': 'IPL 2026 Champion crowned'},
]

# ── build per-match data ──────────────────────────────────
print("Building rosters...")
team_rosters = {t: build_team_roster(t) for t in squads_raw.keys()}

# ── assemble matches ──────────────────────────────────────
def infer_pitch_label(row):
    vals = {'flat': row.get('pitch_flat',0), 'spin': row.get('pitch_spin',0),
            'pace': row.get('pitch_pace',0), 'balanced': row.get('pitch_balanced',1)}
    return max(vals, key=vals.get)

matches_out = []
for i, pred_row in preds.iterrows():
    fx_rows = fixtures[fixtures['match_id'] == pred_row['match_id']]
    fx = fx_rows.iloc[0] if len(fx_rows) else None
    er_rows = enrich[enrich['match_id'] == pred_row['match_id']]
    er = er_rows.iloc[0] if len(er_rows) else None

    t1 = pred_row['team1']; t2 = pred_row['team2']
    time_ist = str(fx['time_ist']) if fx is not None else '19:30'
    city     = str(fx['city'])     if fx is not None else 'Mumbai'

    # Weather defaults from city + time
    wx = get_weather_defaults(city, time_ist)

    # Pitch
    pitch_label = infer_pitch_label(er) if er is not None else 'balanced'

    # Playing 11s
    r1 = team_rosters.get(t1, [])
    r2 = team_rosters.get(t2, [])
    p11_t1 = pick_playing_11(r1)
    p11_t2 = pick_playing_11(r2)
    sub_t1 = pick_impact_sub(r1, p11_t1)
    sub_t2 = pick_impact_sub(r2, p11_t2)

    # Simulate scorecards (2 innings)
    flat_idx = float(er['flatness_index']) if er is not None else 1.0
    inn1 = simulate_scorecard(p11_t1, p11_t2, venue_fl=flat_idx)
    inn2 = simulate_scorecard(p11_t2, p11_t1, target=inn1['total'], venue_fl=flat_idx)

    # Match result
    t1_won_sim = inn1['total'] > inn2['total']
    margin = abs(inn1['total'] - inn2['total'])
    if t1_won_sim:
        result_str = f"{t1} won by {margin} runs"
    else:
        wickets_left = 11 - inn2['wickets']
        result_str = f"{t2} won by {max(1, wickets_left)} wickets"

    matches_out.append({
        'match_id':    str(pred_row['match_id']),
        'date':        str(pred_row['date']),
        'team1':       t1, 'team2': t2,
        'venue':       str(fx['venue']) if fx is not None else pred_row['venue'],
        'city':        city,
        'time_ist':    time_ist,
        # Weather defaults (auto from time+city)
        'wx_defaults': wx,
        # Pitch
        'pitch_label': pitch_label,
        'flatness':    round(float(er['flatness_index']) if er is not None else 1.0, 4),
        'spin_score':  round(float(er['spin_score'])     if er is not None else 0.15, 4),
        'pace_score':  round(float(er['pace_score'])     if er is not None else 1.0, 4),
        'dew_factor':  round(float(er['dew_factor'])     if er is not None else 0.05, 4),
        # Base predictions
        'p_t1_avg':  float(pred_row['p_t1_avg']),
        'p_t1_xgb':  float(pred_row['p_t1_xgb']),
        'p_t1_lgb':  float(pred_row['p_t1_lgb']),
        'p_t1_rf':   float(pred_row['p_t1_rf']),
        'p_t1_mlp':  float(pred_row['p_t1_mlp']),
        # ELO
        't1_elo':    float(er['t1_elo']) if er is not None else 1500,
        't2_elo':    float(er['t2_elo']) if er is not None else 1500,
        # Form
        't1_batting': float(er['t1_batting_score']) if er is not None else 30.0,
        't2_batting': float(er['t2_batting_score']) if er is not None else 30.0,
        't1_bowling': float(er['t1_bowling_score']) if er is not None else 0.4,
        't2_bowling': float(er['t2_bowling_score']) if er is not None else 0.4,
        # Playing 11
        'p11_t1': p11_t1,
        'p11_t2': p11_t2,
        'sub_t1': sub_t1,
        'sub_t2': sub_t2,
        # Scorecards
        'inn1': {**inn1, 'batting_team': t1, 'bowling_team': t2},
        'inn2': {**inn2, 'batting_team': t2, 'bowling_team': t1},
        'sim_result': result_str,
    })

print(f"Built {len(matches_out)} matches with scorecards")

# ── final JSON bundle ─────────────────────────────────────
dashboard_data = {
    'matches':       matches_out,
    'mc':            mc,
    'elo':           elo_cur,
    'metrics':       metrics['test_metrics'],
    'lookup':        lookup,
    'points_table':  final_table_rows,
    'bracket':       bracket,
}

out_path = DATA_DIR / 'dashboard_data_v2.json'
with open(out_path, 'w') as f:
    json.dump(dashboard_data, f)

sz = out_path.stat().st_size / 1024
print(f"\n✓ dashboard_data_v2.json  {sz:.1f} KB")
print(f"  Matches: {len(matches_out)}")
print(f"  Points table top 4: {[r['team'] for r in final_table_rows[:4]]}")
print(f"  Bracket: {[b['match'] for b in bracket]}")
print(f"\nSample inn1: {matches_out[0]['inn1']['total']} runs / {matches_out[0]['inn1']['wickets']} wkts")
print(f"Sample p11_t1: {[p['name'] for p in matches_out[0]['p11_t1'][:4]]}")
print(f"Sample sub: {matches_out[0]['sub_t1']}")
