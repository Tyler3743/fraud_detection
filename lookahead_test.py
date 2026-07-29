import time
import numpy as np
import pandas as pd
from snapml import GraphFeaturePreprocessor
from xgboost import XGBClassifier

from metrics import evaluate, find_best_threshold
# wsl, cd /mnt/d/ct551_v2, source ~/venv_gfp/bin/activate, python lookahead_test.py
CSV = 'dataset_high/HI-Small_Trans_split_index.csv'
N = None           
BATCHES = [1, 128, 2048]
SEEDS = [0, 1, 2]

# cấu hình theo GFP
BINS = [2, 4, 8, 16, 32, 64]
CFG = {
    'num_threads': 4, 'time_window': 86400,
    'vertex_stats': True, 'vertex_stats_tw': 86400, 'vertex_stats_cols': [4],
    'fan': True, 'fan_tw': 86400, 'fan_bins': BINS,
    'degree': True, 'degree_tw': 86400, 'degree_bins': BINS,
    'scatter-gather': True, 'scatter-gather_tw': 21600, 'scatter-gather_bins': BINS,
    'temp-cycle': True, 'temp-cycle_tw': 86400, 'temp-cycle_bins': BINS,
    'lc-cycle': True, 'lc-cycle_tw': 86400, 'lc-cycle_bins': BINS, 'lc-cycle_len': 10,
}
df = pd.read_csv(CSV, nrows=N,
                 dtype={'From Bank': str, 'Account': str,
                        'To Bank': str, 'Account.1': str})
ts = (pd.to_datetime(df['Timestamp']).astype('int64') // 10**9).to_numpy()

assert (np.diff(ts) >= 0).all(), 'Stream chua sort theo thoi gian!'

src = (df['From Bank'] + '_' + df['Account']).astype('category').cat.codes
dst = (df['To Bank'] + '_' + df['Account.1']).astype('category').cat.codes
X = np.column_stack([np.arange(len(df)), src, dst, ts,
                     df['Amount Paid']]).astype(np.float64)

y = df['Is Laundering'].to_numpy()
split = df['split'].to_numpy()
tr, va, te = split == 'train', split == 'val', split == 'test'
print(f'{len(df):,} tx | train {tr.sum():,} val {va.sum():,} test {te.sum():,} '
      f'| pos {y.sum():,} ({100*y.mean():.3f}%)')


def exposure_rate(ts, B):
    if B == 1:
        return 0.0
    hit = 0
    for i in range(0, len(ts), B):
        w = ts[i:i + B]
        hit += (w < w.max()).sum()
    return 100 * hit / len(ts)


def run_gfp(B):
    g = GraphFeaturePreprocessor()
    g.set_params(CFG)
    out = None
    t0 = time.time()
    for i in range(0, len(X), B):
        o = g.transform(X[i:i + B])
        if out is None:
            out = np.empty((len(X), o.shape[1]), dtype=np.float32)
        out[i:i + B] = o
    return out, time.time() - t0


def fit_eval(F, seed):
    """XGBoost giong het nhau cho moi B; threshold tune tren val -> ap sang test."""
    m = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8,
                      tree_method='hist', eval_metric='aucpr', n_jobs=4,
                      scale_pos_weight=(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1),
                      early_stopping_rounds=30, random_state=seed)
    m.fit(F[tr], y[tr], eval_set=[(F[va], y[va])], verbose=False)
    s_va, s_te = m.predict_proba(F[va])[:, 1], m.predict_proba(F[te])[:, 1]
    thr = find_best_threshold(y[va], s_va)
    return evaluate(y[te], s_te, thr)


rows, ref = [], None
for B in BATCHES:
    F, dt = run_gfp(B)
    F = F[:, 4:]                      
    exp = exposure_rate(ts, B)
    if ref is None:
        ref = F[te].copy()            
        blur = 0.0
    else:
        blur = 100 * (~np.isclose(F[te], ref, equal_nan=True)).any(axis=1).mean()
    for seed in SEEDS:
        r = fit_eval(F, seed)
        rows.append(dict(B=B, seed=seed, f1=r['f1_minority'], pr_auc=r['pr_auc'],
                         precision=r['precision'], recall=r['recall'],
                         exposure=exp, blur=blur, gfp_s=dt))
        print(f"B={B:5d} seed={seed} f1={r['f1_minority']:.4f} "
              f"pr_auc={r['pr_auc']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}")
    del F

res = pd.DataFrame(rows)
agg = res.groupby('B').agg(f1_mean=('f1', 'mean'), f1_std=('f1', 'std'),
                           pr_auc_mean=('pr_auc', 'mean'),
                           exposure=('exposure', 'first'), blur=('blur', 'first'),
                           gfp_s=('gfp_s', 'first'))
base = agg['f1_mean'].loc[BATCHES[0]]
agg['delta_f1'] = agg['f1_mean'] - base
print(agg.to_string(float_format=lambda v: f'{v:.4f}'))
res.to_csv('lookahead_batch_curve.csv', index=False)                                                                                