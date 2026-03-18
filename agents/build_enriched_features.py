"""
Phase 4: Build Enriched Feature Matrix
Joins all data sources into a unified ~33-column feature matrix.

Sources:
  - match_features.csv       : 13 original rolling/form features
  - elo_features.csv         : 4 ELO features (joined on match_id)
  - weather_features.csv     : 6 weather features (joined on match_id)
  - venue_pitch_profiles.csv : 4 pitch features (joined on venue name)
  - player_form_features.csv : 8 player form features (2026 predictions only)

Output:
  - enriched_features.csv    : full historical training matrix
  - enriched_features_2026.csv : IPL 2026 prediction rows with player form
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path

DATA_DIR = Path('/sessions/peaceful-cool-keller/ipl_data')

# ─────────────────────────────────────────────
# 1. Load all source files
# ─────────────────────────────────────────────
print("Loading source files...")

mf   = pd.read_csv(DATA_DIR / 'match_features.csv')
elo  = pd.read_csv(DATA_DIR / 'elo_features.csv')
wx   = pd.read_csv(DATA_DIR / 'weather_features.csv')
vp   = pd.read_csv(DATA_DIR / 'venue_pitch_profiles.csv')
pf   = pd.read_csv(DATA_DIR / 'player_form_features.csv')
fix  = pd.read_csv(DATA_DIR / 'ipl_2026_fixtures.csv')
mc   = pd.read_csv(DATA_DIR / 'matches_clean.csv')

print(f"  match_features  : {mf.shape}")
print(f"  elo_features    : {elo.shape}")
print(f"  weather_features: {wx.shape}")
print(f"  venue_pitch     : {vp.shape}")
print(f"  player_form     : {pf.shape}")
print(f"  fixtures_2026   : {fix.shape}")

# ─────────────────────────────────────────────
# 2. Venue name normalisation
#    Match the messy strings from match_features to the venue_pitch keys
# ─────────────────────────────────────────────
def normalise_venue(v: str) -> str:
    """Lower-case, strip punctuation and common suffixes for fuzzy matching."""
    v = str(v).lower()
    v = re.sub(r'[.,\-]', ' ', v)
    # Drop city suffixes that appear in some names but not others
    for suffix in [', uppal', ', mumbai', ', hyderabad', ', delhi',
                   ', bangalore', ', chennai', ', kolkata', ', jaipur',
                   ', ahmedabad', ', pune', ', mohali', ', dharamsala',
                   ', lucknow', ', guwahati', ', raipur']:
        v = v.replace(suffix, '')
    v = re.sub(r'\s+', ' ', v).strip()
    return v

vp['venue_norm'] = vp['venue'].apply(normalise_venue)
mf['venue_norm'] = mf['venue'].apply(normalise_venue)

# Build exact-match lookup first
venue_map = dict(zip(vp['venue_norm'], vp['venue']))

# For any still-unmapped venues in mf, do longest-common-token matching
mf_venues_unmapped = set(mf['venue_norm']) - set(vp['venue_norm'])
if mf_venues_unmapped:
    print(f"\n  Venue fuzzy-matching {len(mf_venues_unmapped)} unresolved venues...")
    vp_norms = vp['venue_norm'].tolist()
    for mv in sorted(mf_venues_unmapped):
        mv_tokens = set(mv.split())
        best_score, best_match = 0, None
        for vn in vp_norms:
            vn_tokens = set(vn.split())
            score = len(mv_tokens & vn_tokens) / max(len(mv_tokens | vn_tokens), 1)
            if score > best_score:
                best_score, best_match = score, vn
        if best_score >= 0.4:
            venue_map[mv] = vp[vp['venue_norm'] == best_match]['venue'].values[0]
            print(f"    '{mv}' → '{best_match}' (score={best_score:.2f})")
        else:
            print(f"    UNMATCHED: '{mv}' (best={best_match}, score={best_score:.2f})")

mf['venue_key'] = mf['venue_norm'].map(venue_map)

# Attach pitch features via venue_key
vp_cols = ['venue', 'flatness_index', 'pace_score', 'spin_score', 'dew_factor', 'pitch_label']
vp_merge = vp[vp_cols].rename(columns={'venue': 'venue_key'})
mf = mf.merge(vp_merge, on='venue_key', how='left')

matched_pitch = mf['flatness_index'].notna().sum()
print(f"\n  Pitch features attached: {matched_pitch}/{len(mf)} matches "
      f"({matched_pitch/len(mf)*100:.1f}%)")

# ─────────────────────────────────────────────
# 3. Join ELO features  (on match_id, perfect 1:1)
# ─────────────────────────────────────────────
elo_cols = ['match_id', 't1_elo', 't2_elo', 'elo_diff', 'elo_win_prob_t1']
mf = mf.merge(elo[elo_cols], on='match_id', how='left')
print(f"\n  ELO features attached: {mf['t1_elo'].notna().sum()}/{len(mf)}")

# ─────────────────────────────────────────────
# 4. Join Weather features  (on match_id)
# ─────────────────────────────────────────────
wx_cols = ['match_id', 'temp_avg', 'precipitation', 'windspeed',
           'humidity_evening', 'dewpoint_evening', 'rain_risk', 'dew_risk']
mf = mf.merge(wx[wx_cols], on='match_id', how='left')

wx_matched = mf['temp_avg'].notna().sum()
print(f"  Weather features attached: {wx_matched}/{len(mf)}")

# Fill missing weather with per-city median (approx climate baseline)
# We'll use matches_clean to get city info
mc_city = mc[['match_id', 'city']].drop_duplicates()
mf = mf.merge(mc_city, on='match_id', how='left')

wx_numeric = ['temp_avg', 'precipitation', 'windspeed',
              'humidity_evening', 'dewpoint_evening']
for col in wx_numeric:
    city_median = mf.groupby('city')[col].transform('median')
    mf[col] = mf[col].fillna(city_median)
    # Fallback to global median if city still missing
    mf[col] = mf[col].fillna(mf[col].median())

# Binary weather flags
mf['rain_risk'] = mf['rain_risk'].fillna(0).astype(int)
mf['dew_risk']  = mf['dew_risk'].fillna(0).astype(int)

# Fill missing pitch metrics with global medians
for col in ['flatness_index', 'pace_score', 'spin_score', 'dew_factor']:
    mf[col] = mf[col].fillna(mf[col].median())

mf['pitch_label'] = mf['pitch_label'].fillna('balanced')

print(f"\n  After imputation — NaN counts in key cols:")
key_cols = ['t1_elo', 'elo_diff', 'temp_avg', 'humidity_evening',
            'flatness_index', 'spin_score']
for c in key_cols:
    print(f"    {c}: {mf[c].isna().sum()} NaN")

# ─────────────────────────────────────────────
# 5. Encode pitch_label as one-hot
# ─────────────────────────────────────────────
pitch_dummies = pd.get_dummies(mf['pitch_label'], prefix='pitch').astype(int)
# Ensure all 4 categories always exist
for cat in ['pitch_flat', 'pitch_spin', 'pitch_pace', 'pitch_balanced']:
    if cat not in pitch_dummies.columns:
        pitch_dummies[cat] = 0
mf = pd.concat([mf, pitch_dummies], axis=1)

# ─────────────────────────────────────────────
# 6. Define final ENRICHED feature columns
# ─────────────────────────────────────────────
ORIGINAL_FEATURES = [
    't1_overall_wr', 't2_overall_wr',
    't1_recent_wr', 't2_recent_wr',
    't1_form5', 't2_form5',
    'h2h_t1_wr',
    't1_venue_wr', 't2_venue_wr',
    'toss_winner_is_t1',
]

ELO_FEATURES = [
    't1_elo', 't2_elo', 'elo_diff', 'elo_win_prob_t1',
]

WEATHER_FEATURES = [
    'temp_avg', 'precipitation', 'windspeed',
    'humidity_evening', 'dewpoint_evening',
    'rain_risk', 'dew_risk',
]

PITCH_FEATURES = [
    'flatness_index', 'pace_score', 'spin_score', 'dew_factor',
    'pitch_flat', 'pitch_spin', 'pitch_pace', 'pitch_balanced',
]

ALL_FEATURE_COLS = ORIGINAL_FEATURES + ELO_FEATURES + WEATHER_FEATURES + PITCH_FEATURES
TARGET_COL = 'team1_won'

print(f"\n  Total feature columns: {len(ALL_FEATURE_COLS)}")
print(f"  Feature groups: {len(ORIGINAL_FEATURES)} original + "
      f"{len(ELO_FEATURES)} ELO + {len(WEATHER_FEATURES)} weather + "
      f"{len(PITCH_FEATURES)} pitch")

# ─────────────────────────────────────────────
# 7. Build clean training-ready dataframe
# ─────────────────────────────────────────────
train_df = mf[['match_id', 'date', 'season', 'team1', 'team2', 'venue'] +
              ALL_FEATURE_COLS + [TARGET_COL]].copy()

# Drop rows where target is missing
train_df = train_df.dropna(subset=[TARGET_COL])
train_df[TARGET_COL] = train_df[TARGET_COL].astype(int)

# Drop rows with any NaN in feature columns
rows_before = len(train_df)
train_df = train_df.dropna(subset=ALL_FEATURE_COLS)
print(f"\n  Rows after dropna on features: {len(train_df)} (dropped {rows_before - len(train_df)})")

# Ensure numeric types
for col in ALL_FEATURE_COLS:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

print(f"\n  Final enriched training set: {train_df.shape}")
print(f"  Class balance — team1_won=1: {train_df[TARGET_COL].mean():.3f}")

# Save
train_df.to_csv(DATA_DIR / 'enriched_features.csv', index=False)
print(f"\n  ✓ Saved enriched_features.csv")

# ─────────────────────────────────────────────
# 8. Build IPL 2026 prediction rows
#    (same features + player form for current teams)
# ─────────────────────────────────────────────
print("\n--- Building IPL 2026 prediction feature rows ---")

# Load current ELO ratings
with open(DATA_DIR / 'elo_current.json') as f:
    elo_current = json.load(f)

print("  Current ELO ratings:")
for t, r in sorted(elo_current.items(), key=lambda x: -x[1]):
    print(f"    {t}: {r:.1f}")

# Player form lookup
pf_lookup = pf.set_index('team')

def get_elo(team):
    return elo_current.get(team, 1500.0)

def expected_win_elo(r1, r2):
    return 1.0 / (1 + 10 ** ((r2 - r1) / 400))

def get_player_form(team, prefix):
    if team in pf_lookup.index:
        row = pf_lookup.loc[team]
        return {
            f'{prefix}_batting_score': row['team_batting_score'],
            f'{prefix}_bowling_score': row['team_bowling_score'],
            f'{prefix}_batting_sr':    row['team_batting_sr'],
            f'{prefix}_bowling_econ':  row['team_bowling_econ'],
        }
    else:
        return {
            f'{prefix}_batting_score': pf['team_batting_score'].mean(),
            f'{prefix}_bowling_score': pf['team_bowling_score'].mean(),
            f'{prefix}_batting_sr':    pf['team_batting_sr'].mean(),
            f'{prefix}_bowling_econ':  pf['team_bowling_econ'].mean(),
        }

rows_2026 = []
# Use recent team stats as proxy rolling features for 2026
# (compute from historical data for each IPL 2026 team)
recent_stats = {}
for _, row in train_df[train_df['season'] == train_df['season'].max()].iterrows():
    for team, prefix in [(row['team1'], 't1'), (row['team2'], 't2')]:
        if team not in recent_stats:
            recent_stats[team] = {
                'overall_wr': row[f'{prefix}_overall_wr'],
                'recent_wr':  row[f'{prefix}_recent_wr'],
                'form5':      row[f'{prefix}_form5'],
            }

# Global fallback values from training data
global_mean = train_df[ALL_FEATURE_COLS].mean()

for _, fixture in fix.iterrows():
    t1 = fixture.get('team1', fixture.get('Team1', ''))
    t2 = fixture.get('team2', fixture.get('Team2', ''))
    venue = fixture.get('venue', fixture.get('Venue', ''))
    date  = fixture.get('date', fixture.get('Date', ''))

    # ELO
    r1 = get_elo(t1)
    r2 = get_elo(t2)
    elo_diff = r1 - r2
    elo_prob = expected_win_elo(r1, r2)

    # Pitch features for this venue
    venue_norm = normalise_venue(str(venue))
    venue_key  = venue_map.get(venue_norm)
    if venue_key:
        vrow = vp[vp['venue'] == venue_key].iloc[0]
        fi   = vrow['flatness_index']
        ps   = vrow['pace_score']
        ss   = vrow['spin_score']
        df_  = vrow['dew_factor']
        plbl = vrow['pitch_label']
    else:
        fi, ps, ss, df_, plbl = (
            global_mean['flatness_index'], global_mean['pace_score'],
            global_mean['spin_score'], global_mean['dew_factor'], 'balanced'
        )

    # Pitch one-hot
    pitch_flat     = int(plbl == 'flat')
    pitch_spin     = int(plbl == 'spin')
    pitch_pace     = int(plbl == 'pace')
    pitch_balanced = int(plbl == 'balanced')

    # Rolling team stats — use latest available or global mean
    def stat(team, key):
        return recent_stats.get(team, {}).get(key, global_mean.get(f't1_{key}', 0.5))

    row_dict = {
        'match_id': fixture.get('match_id', fixture.get('MatchID', f'2026_{_}')),
        'date':  date,
        'team1': t1,
        'team2': t2,
        'venue': venue,
        # Original rolling features
        't1_overall_wr':   stat(t1, 'overall_wr'),
        't2_overall_wr':   stat(t2, 'overall_wr'),
        't1_recent_wr':    stat(t1, 'recent_wr'),
        't2_recent_wr':    stat(t2, 'recent_wr'),
        't1_form5':        stat(t1, 'form5'),
        't2_form5':        stat(t2, 'form5'),
        'h2h_t1_wr':       0.5,   # will be updated from historical H2H
        't1_venue_wr':     0.5,
        't2_venue_wr':     0.5,
        'toss_winner_is_t1': 0,   # unknown pre-match
        # ELO
        't1_elo':          r1,
        't2_elo':          r2,
        'elo_diff':        elo_diff,
        'elo_win_prob_t1': elo_prob,
        # Weather (use venue/season climate baseline)
        'temp_avg':        global_mean['temp_avg'],
        'precipitation':   0.0,
        'windspeed':       global_mean['windspeed'],
        'humidity_evening': global_mean['humidity_evening'],
        'dewpoint_evening': global_mean['dewpoint_evening'],
        'rain_risk':        0,
        'dew_risk':         0,
        # Pitch
        'flatness_index':  fi,
        'pace_score':      ps,
        'spin_score':      ss,
        'dew_factor':      df_,
        'pitch_flat':      pitch_flat,
        'pitch_spin':      pitch_spin,
        'pitch_pace':      pitch_pace,
        'pitch_balanced':  pitch_balanced,
        # Player form (2026-specific)
        **get_player_form(t1, 't1'),
        **get_player_form(t2, 't2'),
    }
    rows_2026.append(row_dict)

fix_2026 = pd.DataFrame(rows_2026)
fix_2026.to_csv(DATA_DIR / 'enriched_features_2026.csv', index=False)
print(f"\n  ✓ Saved enriched_features_2026.csv  shape={fix_2026.shape}")

# ─────────────────────────────────────────────
# 9. Summary
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 4 COMPLETE — Feature Matrix Summary")
print("="*60)
print(f"  Historical training rows : {len(train_df)}")
print(f"  IPL 2026 prediction rows : {len(fix_2026)}")
print(f"\n  Feature columns ({len(ALL_FEATURE_COLS)} total):")
groups = [
    ("Original rolling", ORIGINAL_FEATURES),
    ("ELO",             ELO_FEATURES),
    ("Weather",         WEATHER_FEATURES),
    ("Pitch",           PITCH_FEATURES),
]
for gname, gcols in groups:
    print(f"  [{gname}] {gcols}")

print(f"\n  Class balance (team1_won=1): {train_df[TARGET_COL].mean():.3f}")
print(f"\n  train_df dtypes:")
print(train_df[ALL_FEATURE_COLS].dtypes.to_string())

# Feature correlation with target
corr = train_df[ALL_FEATURE_COLS + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
print(f"\n  Top 10 features by |correlation| with team1_won:")
print(corr.abs().sort_values(ascending=False).head(10).to_string())

# Save the feature column list for Phase 5
feature_meta = {
    'all_features': ALL_FEATURE_COLS,
    'original':     ORIGINAL_FEATURES,
    'elo':          ELO_FEATURES,
    'weather':      WEATHER_FEATURES,
    'pitch':        PITCH_FEATURES,
    'target':       TARGET_COL,
    'n_train':      len(train_df),
    'n_2026':       len(fix_2026),
}
with open(DATA_DIR / 'feature_meta.json', 'w') as f:
    json.dump(feature_meta, f, indent=2)
print("\n  ✓ Saved feature_meta.json")
