"""
Phase 5: Train Full Model Suite + Stacking Meta-Learner

Models:
  1. XGBoost
  2. LightGBM
  3. Random Forest
  4. PyTorch MLP  (4-layer: input→64→128→64→1 with BatchNorm + Dropout)
  5. Stacking Meta-Learner (LogisticRegression on OOF predictions from 1-4)

Train/test split: 2008-2023 train | 2024-2025 test  (temporal, no leakage)
Outputs:
  - models_v2.pkl          : all trained model objects
  - model_metrics_v2.json  : accuracy + AUC per model
  - oof_predictions.csv    : out-of-fold predictions used for stacking
  - ipl_2026_predictions_v2.csv : updated 2026 fixture probabilities
"""

import json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

DATA_DIR = Path('/sessions/peaceful-cool-keller/ipl_data')

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
print("Loading enriched feature matrix...")
df = pd.read_csv(DATA_DIR / 'enriched_features.csv')
df['date'] = pd.to_datetime(df['date'])

with open(DATA_DIR / 'feature_meta.json') as f:
    meta = json.load(f)

FEATURE_COLS = meta['all_features']
TARGET       = meta['target']

print(f"  Total rows: {len(df)}, Features: {len(FEATURE_COLS)}")

# Temporal split
train_df = df[df['date'].dt.year <= 2023].copy()
test_df  = df[df['date'].dt.year >= 2024].copy()

X_train = train_df[FEATURE_COLS].values.astype(np.float32)
y_train = train_df[TARGET].values.astype(np.float32)
X_test  = test_df[FEATURE_COLS].values.astype(np.float32)
y_test  = test_df[TARGET].values.astype(np.float32)

print(f"  Train: {X_train.shape}  (years ≤ 2023)")
print(f"  Test:  {X_test.shape}   (years 2024-2025)")
print(f"  Class balance — train: {y_train.mean():.3f}  test: {y_test.mean():.3f}")

# Scale for MLP + Logistic Regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
X_test_sc  = scaler.transform(X_test).astype(np.float32)

# ─────────────────────────────────────────────
# 2. PyTorch MLP definition
# ─────────────────────────────────────────────
class CricketMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

def train_mlp(X_tr, y_tr, X_val, y_val, n_epochs=60, lr=0.001, batch_size=64, verbose=True):
    """Train MLP and return trained model + val predictions."""
    device = torch.device('cpu')
    model  = CricketMLP(X_tr.shape[1]).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)
    loss_fn = nn.BCELoss()

    ds     = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_state    = None
    patience_cnt  = 0

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb.to(device))
            loss = loss_fn(pred, yb.to(device))
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(X_val).to(device)).numpy()
            val_loss = loss_fn(torch.tensor(val_pred), torch.tensor(y_val)).item()
            sched.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 12:
                if verbose:
                    print(f"    Early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_proba = model(torch.tensor(X_val).to(device)).numpy()
    return model, val_proba

# ─────────────────────────────────────────────
# 3. Out-of-fold stacking infrastructure
# ─────────────────────────────────────────────
N_SPLITS   = 5
skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# OOF arrays for meta-learner
oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_rf  = np.zeros(len(X_train))
oof_mlp = np.zeros(len(X_train))

# Test predictions (averaged over folds)
test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_rf  = np.zeros(len(X_test))
test_mlp = np.zeros(len(X_test))

print("\n--- Cross-validation (5-fold OOF for stacking) ---")

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    Xf_tr, Xf_val = X_train[tr_idx], X_train[val_idx]
    yf_tr, yf_val = y_train[tr_idx], y_train[val_idx]
    Xf_tr_sc      = X_train_sc[tr_idx]
    Xf_val_sc     = X_train_sc[val_idx]

    print(f"\n  Fold {fold+1}/{N_SPLITS}  (train={len(tr_idx)}, val={len(val_idx)})")

    # --- XGBoost ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_model.fit(Xf_tr, yf_tr,
                  eval_set=[(Xf_val, yf_val)],
                  verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(Xf_val)[:, 1]
    test_xgb        += xgb_model.predict_proba(X_test)[:, 1] / N_SPLITS
    acc = accuracy_score(yf_val, (oof_xgb[val_idx] > 0.5).astype(int))
    print(f"    XGB   val_acc={acc:.4f}")

    # --- LightGBM ---
    lgb_model = lgb.LGBMClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    lgb_model.fit(Xf_tr, yf_tr,
                  eval_set=[(Xf_val, yf_val)],
                  callbacks=[lgb.early_stopping(40, verbose=False),
                             lgb.log_evaluation(-1)])
    oof_lgb[val_idx] = lgb_model.predict_proba(Xf_val)[:, 1]
    test_lgb        += lgb_model.predict_proba(X_test)[:, 1] / N_SPLITS
    acc = accuracy_score(yf_val, (oof_lgb[val_idx] > 0.5).astype(int))
    print(f"    LGB   val_acc={acc:.4f}")

    # --- Random Forest ---
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    rf_model.fit(Xf_tr, yf_tr)
    oof_rf[val_idx] = rf_model.predict_proba(Xf_val)[:, 1]
    test_rf        += rf_model.predict_proba(X_test)[:, 1] / N_SPLITS
    acc = accuracy_score(yf_val, (oof_rf[val_idx] > 0.5).astype(int))
    print(f"    RF    val_acc={acc:.4f}")

    # --- PyTorch MLP ---
    print("    MLP   training...", end='', flush=True)
    mlp_model, oof_mlp[val_idx] = train_mlp(
        Xf_tr_sc, yf_tr, Xf_val_sc, yf_val,
        n_epochs=60, verbose=False
    )
    test_mlp_fold = []
    mlp_model.eval()
    with torch.no_grad():
        test_mlp_fold = mlp_model(torch.tensor(X_test_sc)).numpy()
    test_mlp += test_mlp_fold / N_SPLITS
    acc = accuracy_score(yf_val, (oof_mlp[val_idx] > 0.5).astype(int))
    print(f" val_acc={acc:.4f}")

# ─────────────────────────────────────────────
# 4. OOF performance summary
# ─────────────────────────────────────────────
print("\n--- OOF Performance (full training set) ---")
oof_results = {}
for name, oof in [('XGBoost', oof_xgb), ('LightGBM', oof_lgb),
                  ('RandomForest', oof_rf), ('MLP', oof_mlp)]:
    acc = accuracy_score(y_train, (oof > 0.5).astype(int))
    auc = roc_auc_score(y_train, oof)
    oof_results[name] = {'oof_acc': acc, 'oof_auc': auc}
    print(f"  {name:<14} OOF acc={acc:.4f}  AUC={auc:.4f}")

# ─────────────────────────────────────────────
# 5. Train Stacking Meta-Learner on OOF predictions
# ─────────────────────────────────────────────
print("\n--- Training Stacking Meta-Learner ---")
X_meta_train = np.column_stack([oof_xgb, oof_lgb, oof_rf, oof_mlp])
X_meta_test  = np.column_stack([test_xgb, test_lgb, test_rf, test_mlp])

meta_scaler = StandardScaler()
X_meta_train_sc = meta_scaler.fit_transform(X_meta_train)
X_meta_test_sc  = meta_scaler.transform(X_meta_test)

stack_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
stack_model.fit(X_meta_train_sc, y_train)

stack_coefs = dict(zip(['XGB', 'LGB', 'RF', 'MLP'], stack_model.coef_[0]))
print(f"  Meta-learner coefficients: {stack_coefs}")

# ─────────────────────────────────────────────
# 6. Retrain base models on FULL training set
# ─────────────────────────────────────────────
print("\n--- Retraining base models on full training set ---")

xgb_final = xgb.XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='logloss',
    random_state=42, n_jobs=-1, verbosity=0,
)
xgb_final.fit(X_train, y_train, verbose=False)
print("  XGB done")

lgb_final = lgb.LGBMClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=-1,
)
lgb_final.fit(X_train, y_train)
print("  LGB done")

rf_final = RandomForestClassifier(
    n_estimators=400, max_depth=8, min_samples_leaf=5,
    random_state=42, n_jobs=-1,
)
rf_final.fit(X_train, y_train)
print("  RF done")

print("  MLP (full train)...", end='', flush=True)
mlp_final, _ = train_mlp(X_train_sc, y_train, X_test_sc, y_test, n_epochs=80, verbose=False)
print(" done")

# ─────────────────────────────────────────────
# 7. Test set evaluation
# ─────────────────────────────────────────────
print("\n--- Test Set Performance (2024-2025) ---")

def get_proba(model, X, is_mlp=False, is_torch=False):
    if is_torch:
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(X)).numpy()
    return model.predict_proba(X)[:, 1]

p_xgb   = get_proba(xgb_final, X_test)
p_lgb   = get_proba(lgb_final, X_test)
p_rf    = get_proba(rf_final, X_test)
p_mlp   = get_proba(mlp_final, X_test_sc, is_torch=True)

X_meta_test_final = np.column_stack([p_xgb, p_lgb, p_rf, p_mlp])
X_meta_test_final_sc = meta_scaler.transform(X_meta_test_final)
p_stack = stack_model.predict_proba(X_meta_test_final_sc)[:, 1]

# Simple average ensemble (comparison)
p_avg = (p_xgb + p_lgb + p_rf + p_mlp) / 4.0

metrics = {}
for name, proba in [('XGBoost', p_xgb), ('LightGBM', p_lgb),
                    ('RandomForest', p_rf), ('MLP', p_mlp),
                    ('SimpleEnsemble', p_avg), ('StackingMeta', p_stack)]:
    acc = accuracy_score(y_test, (proba > 0.5).astype(int))
    auc = roc_auc_score(y_test, proba)
    metrics[name] = {'accuracy': round(acc, 4), 'auc_roc': round(auc, 4)}
    tag = " ◀ BEST" if name in ('StackingMeta', 'SimpleEnsemble') else ""
    print(f"  {name:<18} acc={acc:.4f}  AUC={auc:.4f}{tag}")

# ─────────────────────────────────────────────
# 8. Feature importances (XGB)
# ─────────────────────────────────────────────
print("\n--- Top 15 Feature Importances (XGBoost) ---")
feat_imp = pd.Series(xgb_final.feature_importances_, index=FEATURE_COLS)
print(feat_imp.sort_values(ascending=False).head(15).to_string())

# ─────────────────────────────────────────────
# 9. IPL 2026 predictions
# ─────────────────────────────────────────────
print("\n--- IPL 2026 Predictions ---")
fix_2026 = pd.read_csv(DATA_DIR / 'enriched_features_2026.csv')

# Use only the core FEATURE_COLS (no player form — not in training set)
X_2026 = fix_2026[FEATURE_COLS].values.astype(np.float32)
X_2026_sc = scaler.transform(X_2026).astype(np.float32)

p2_xgb  = get_proba(xgb_final, X_2026)
p2_lgb  = get_proba(lgb_final, X_2026)
p2_rf   = get_proba(rf_final,  X_2026)
p2_mlp  = get_proba(mlp_final, X_2026_sc, is_torch=True)

X2_meta = np.column_stack([p2_xgb, p2_lgb, p2_rf, p2_mlp])
X2_meta_sc = meta_scaler.transform(X2_meta)
p2_stack = stack_model.predict_proba(X2_meta_sc)[:, 1]

preds_2026 = fix_2026[['match_id', 'date', 'team1', 'team2', 'venue']].copy()
preds_2026['p_t1_xgb']    = np.round(p2_xgb,  4)
preds_2026['p_t1_lgb']    = np.round(p2_lgb,  4)
preds_2026['p_t1_rf']     = np.round(p2_rf,   4)
preds_2026['p_t1_mlp']    = np.round(p2_mlp,  4)
preds_2026['p_t1_stack']  = np.round(p2_stack, 4)
preds_2026['p_t1_avg']    = np.round((p2_xgb+p2_lgb+p2_rf+p2_mlp)/4, 4)

print(preds_2026[['date', 'team1', 'team2', 'p_t1_stack', 'p_t1_avg']].to_string(index=False))

preds_2026.to_csv(DATA_DIR / 'ipl_2026_predictions_v2.csv', index=False)
print("\n  ✓ Saved ipl_2026_predictions_v2.csv")

# ─────────────────────────────────────────────
# 10. Save all models + metadata
# ─────────────────────────────────────────────
print("\n--- Saving models ---")
model_bundle = {
    'xgb':         xgb_final,
    'lgb':         lgb_final,
    'rf':          rf_final,
    'mlp_state':   mlp_final.state_dict(),   # PyTorch state dict
    'mlp_arch':    {'n_features': len(FEATURE_COLS)},
    'stacking':    stack_model,
    'scaler':      scaler,           # for raw features
    'meta_scaler': meta_scaler,      # for meta-learner input
    'feature_cols': FEATURE_COLS,
}

with open(DATA_DIR / 'models_v2.pkl', 'wb') as f:
    pickle.dump(model_bundle, f)
print("  ✓ Saved models_v2.pkl")

# Save metrics
all_metrics = {
    'test_metrics': metrics,
    'oof_metrics':  oof_results,
    'n_train':      int(len(X_train)),
    'n_test':       int(len(X_test)),
    'n_features':   len(FEATURE_COLS),
    'feature_cols': FEATURE_COLS,
}
with open(DATA_DIR / 'model_metrics_v2.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)
print("  ✓ Saved model_metrics_v2.json")

# Save OOF predictions for posterity
oof_df = train_df[['match_id', 'date', 'team1', 'team2', TARGET]].copy()
oof_df['oof_xgb'] = oof_xgb
oof_df['oof_lgb'] = oof_lgb
oof_df['oof_rf']  = oof_rf
oof_df['oof_mlp'] = oof_mlp
oof_df.to_csv(DATA_DIR / 'oof_predictions.csv', index=False)
print("  ✓ Saved oof_predictions.csv")

print("\n" + "="*60)
print("PHASE 5 COMPLETE")
print("="*60)
best_model = max(metrics, key=lambda k: metrics[k]['accuracy'])
print(f"  Best model: {best_model}")
print(f"  Best accuracy: {metrics[best_model]['accuracy']:.4f}")
print(f"  Best AUC-ROC:  {metrics[best_model]['auc_roc']:.4f}")
