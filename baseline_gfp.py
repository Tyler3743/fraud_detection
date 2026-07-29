import pandas as pd, numpy as np, time
from snapml import GraphFeaturePreprocessor

N = 500_000
df = pd.read_csv('/mnt/d/ct551_v2/dataset_high/HI-Small_Trans.csv', nrows=N,
                 dtype={'From Bank':str,'Account':str,'To Bank':str,'Account.1':str})

df['ts'] = pd.to_datetime(df['Timestamp']).astype('int64')//10**9
src = (df['From Bank']+'_'+df['Account']).astype('category').cat.codes
dst = (df['To Bank']+'_'+df['Account.1']).astype('category').cat.codes
X = np.column_stack([np.arange(len(df)), src, dst, df['ts'],
                     df['Amount Paid']]).astype(np.float64)

for name, cfg in [
    ("fan+degree",   dict(fan=True, degree=True, vertex_stats=True)),
    ("+scatter-gath",dict(fan=True, degree=True, vertex_stats=True,
                          **{'scatter-gather':True})),
    ("+cycles(full)",dict(fan=True, degree=True, vertex_stats=True,
                          **{'scatter-gather':True,'temp-cycle':True,'lc-cycle':True})),
]:
    g = GraphFeaturePreprocessor()
    p = {'num_threads': 4, 'time_window': 86400}; p.update(cfg); g.set_params(p)
    B, t0 = 2048, time.time()
    for i in range(0, len(X), B):
        out = g.transform(X[i:i+B])
    dt = time.time() - t0
    print(f"{name:15s} {len(X)/dt:10,.0f} tx/s  {dt:6.1f}s  feat={out.shape[1]}")