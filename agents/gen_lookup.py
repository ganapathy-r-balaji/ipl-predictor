import pickle, json, numpy as np, pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn

DATA_DIR = Path('/sessions/peaceful-cool-keller/ipl_data')

with open(DATA_DIR / 'feature_meta.json') as f:
    meta = json.load(f)
FEATURE_COLS = meta['all_features']

with open(DATA_DIR / 'models_v2.pkl', 'rb') as f:
    bundle = pickle.load(f)

xgb_m = bundle['xgb']
lgb_m = bundle['lgb']
rf_m  = bundle['rf']
scaler = bundle['scaler']

class CricketMLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x).squeeze(1)

mlp = CricketMLP(len(FEATURE_COLS))
mlp.load_state_dict(bundle['mlp_state'])
mlp.eval()

fix2026 = pd.read_csv(DATA_DIR / 'enriched_features_2026.csv')

TOSS_VALS   = [0, 1]
PITCH_TYPES = ['balanced', 'flat', 'spin', 'pace']
TEMP_MAP    = {'cool': 22.0, 'warm': 30.0, 'hot': 38.0}
HUM_MAP     = {'dry': 50.0, 'moderate': 65.0, 'humid': 80.0}
RAIN_VALS   = [0, 1]
DEW_VALS    = [0, 1]

PITCH_SCORES = {
    'flat':     {'flatness_index': 1.15, 'spin_score': 0.10, 'pace_score': 0.95},
    'spin':     {'flatness_index': 0.95, 'spin_score': 0.22, 'pace_score': 0.90},
    'pace':     {'flatness_index': 0.95, 'spin_score': 0.10, 'pace_score': 1.12},
    'balanced': {'flatness_index': 1.00, 'spin_score': 0.15, 'pace_score': 1.00},
}

def pitch_onehot(ptype):
    return {f'pitch_{t}': int(t == ptype) for t in ['flat','spin','pace','balanced']}

def predict_combo(base, overrides):
    row = {**base, **overrides}
    X   = np.array([[row[c] for c in FEATURE_COLS]], dtype='float32')
    pxgb = float(xgb_m.predict_proba(X)[0,1])
    plgb = float(lgb_m.predict_proba(X)[0,1])
    prf  = float(rf_m.predict_proba(X)[0,1])
    Xsc  = scaler.transform(X).astype('float32')
    with torch.no_grad():
        pmlp = float(mlp(torch.tensor(Xsc)).item())
    avg  = (pxgb + plgb + prf + pmlp) / 4
    return {'xgb': round(pxgb,4), 'lgb': round(plgb,4),
            'rf': round(prf,4), 'mlp': round(pmlp,4), 'avg': round(avg,4)}

lookup = {}
for i, frow in fix2026.iterrows():
    mid   = str(frow.get('match_id', i))
    base  = {c: float(frow[c]) for c in FEATURE_COLS}
    base_dew_factor = base.get('dew_factor', 0.05)
    match_lk = {}
    for toss in TOSS_VALS:
        for ptype in PITCH_TYPES:
            ps = PITCH_SCORES[ptype]
            for tname, tval in TEMP_MAP.items():
                for hname, hval in HUM_MAP.items():
                    for rain in RAIN_VALS:
                        for dew in DEW_VALS:
                            ov = {
                                'toss_winner_is_t1': toss,
                                'temp_avg': tval,
                                'humidity_evening': hval,
                                'dewpoint_evening': hval - 5.0,
                                'rain_risk': rain,
                                'dew_risk': dew,
                                'flatness_index': ps['flatness_index'],
                                'spin_score': ps['spin_score'],
                                'pace_score': ps['pace_score'],
                                'dew_factor': base_dew_factor,
                                **pitch_onehot(ptype),
                            }
                            key = f'{toss}|{ptype}|{tname}|{hname}|{rain}|{dew}'
                            match_lk[key] = predict_combo(base, ov)
    lookup[mid] = match_lk
    print(f'Done {i+1}/20: {frow["team1"]} vs {frow["team2"]}')

with open(DATA_DIR / 'prob_lookup.json', 'w') as f:
    json.dump(lookup, f)

sz = Path(DATA_DIR / 'prob_lookup.json').stat().st_size / 1024
print(f'DONE. prob_lookup.json = {sz:.1f} KB')
