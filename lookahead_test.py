"""Lookahead / batch-size ablation cho GraphFeaturePreprocessor (GFP).

Chay trong WSL:
    wsl -d Ubuntu
    cd /mnt/d/ct551_v2
    source ~/venv_gfp/bin/activate
    python lookahead_test.py --n 200000          # smoke test truoc (bat buoc)
    python lookahead_test.py --n 0               # chay full 5.08M dong
    python lookahead_test.py --n 0 --tw-control  # them bien the control time_window

Ghi chu quan trong cho luan van:
  * Timestamp cua HI-Small chi co do phan giai 1 PHUT. Voi B nho (vd 128),
    p50 batch span = 0 giay -> ca lo nam trong cung mot phut, nen "lookahead"
    o day thuc chat chi la dao thu tu trong cung mot phut, von da tuy tien.
    => B=1 khong phai ground truth tuyet doi, chi la moc tham chieu.
  * lookahead_rate la CAN DUOI 1-hop: chi dem giao dich sau trong lo co CHUNG
    TAI KHOAN (src hoac dst) va nam trong time_window. GFP con khai thac
    scatter-gather / cycle multi-hop nen ro ri thuc te >= con so nay.
    Bo loc time_window trong ham nay hau nhu KHONG rang buoc (batch span p95
    ~ 60s << tw 86400s), nen main va tw_control se cho lookahead_rate giong
    het nhau. Day la ky vong, khong phai ket qua co y nghia.
  * blur dung rtol=0, atol=0 (dem khac-bit) de minh bach: F la float32
    (eps ~ 1.19e-7), moi nguong rtol trung gian deu la lua chon tuy y va lam
    nhoe ket qua. equal_nan=True duoc giu de NaN==NaN khong bi tinh la khac.
  * Split trong CSV la LIEN TUC theo thoi gian (train roi val roi test), nen
    doc nrows=N se ra 100% train. Che do --n lay mau theo ti le tu ca ba vung
    -> smoke test co KHOANG TRONG thoi gian giua cac vung, lookahead_rate cua
    no khong so sanh truc tiep voi lan chay full duoc.
"""
import argparse
import os
import queue
import time
import multiprocessing as mp
from bisect import bisect_right

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from metrics import evaluate, find_best_threshold

DT = {'From Bank': str, 'Account': str, 'To Bank': str, 'Account.1': str}

# ----------------------------------------------------------------------------
# Cau hinh GFP: bam theo mac dinh cua thu vien, KHONG ep bins thu cong.
# Paper (Figure 6) dung bins 2 3 ... >=30 (scatter-gather, temp-cycle) va
# 2 3 ... >=10 (simple cycle) -- day chinh la mac dinh cua GFP.
# vertex_stats_cols = [3, 4] = (timestamp, amount) trong X, dung nhu paper.
# ----------------------------------------------------------------------------
TW = 86400
CFG_BASE = {
    'num_threads': 4,
    'time_window': TW,
    'vertex_stats': True, 'vertex_stats_tw': TW, 'vertex_stats_cols': [3, 4],
    'fan': True, 'fan_tw': TW,
    'degree': True, 'degree_tw': TW,
    'scatter-gather': True, 'scatter-gather_tw': 21600,
    'temp-cycle': True, 'temp-cycle_tw': TW,
    'lc-cycle': True, 'lc-cycle_tw': TW, 'lc-cycle_len': 10,
}

# Bien the control: time_window rat lon -> cua so tia gan nhu khong bao gio cat,
# tach rieng anh huong cua "pruning shift" (t_now = ts lon nhat trong graph).
# Span cua dataset ~ 1.53e6 giay nen 1e7 la du bao phu toan bo.
TW_LARGE = 10 ** 7
CFG_TW_LARGE = dict(CFG_BASE)
CFG_TW_LARGE.update({
    'time_window': TW_LARGE, 'vertex_stats_tw': TW_LARGE, 'fan_tw': TW_LARGE,
    'degree_tw': TW_LARGE, 'scatter-gather_tw': TW_LARGE,
    'temp-cycle_tw': TW_LARGE, 'lc-cycle_tw': TW_LARGE,
})

FX_USD = {
    'US Dollar':         1.0,
    'Euro':              1.1717835,      # n=6059
    'Yuan':              0.14930721,     # n=3689
    'UK Pound':          1.2916559,      # n=988
    'Yen':               0.0094876674,   # n=581
    'Rupee':             0.013615817,    # n=433
    'Canadian Dollar':   0.75797771,     # n=318
    'Mexican Peso':      0.047296756,    # n=162
    'Swiss Franc':       1.0928962,      # n=121
    'Ruble':             0.012852736,    # n=88
    'Australian Dollar': 0.70781423,     # n=79
    'Brazil Real':       0.17710086,     # n=44
    'Saudi Riyal':       0.26658847,     # n=32
    'Shekel':            0.29612073,     # n=24
    'Bitcoin':           11881.504,      # n=8
}
# ----------------------------------------------------------------------------
# Do RAM cua GFP
# ----------------------------------------------------------------------------
def _rss_mb():
    """RSS hien tai (MB), doc tu /proc/self/statm."""
    try:
        with open('/proc/self/statm') as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf('SC_PAGE_SIZE') / 1024 ** 2
    except Exception:
        return float('nan')


def _gfp_worker(cfg, X, B, out_path, q):
    """Chay GFP trong process con, ghi output bang buffered write.

    Do RAM -- hai chi tiet da verify bang thuc nghiem:

    (1) Khi fork, kernel dat hiwater_rss cua con = RSS HIEN TAI luc fork
        (dup_mm: mm->hiwater_rss = get_mm_rss(mm)), khong phai peak cua cha.
        Da do: cha peak 729 MB / RSS hien tai 179 MB -> con bao ru_maxrss
        170 MB. Nghia la con KHONG lan peak cua cha, NHUNG van lan toan bo
        RSS ke thua (df, X, ts, src, dst...). Vi vay phai tru rss0. Lenh ghi
        '5' vao /proc/self/clear_refs lam dung viec kernel vua lam, nen la
        thua; giu lai chi nhu lop phong ve neu chay tren nhan khac.

    (2) Output ghi bang f.write() chu KHONG dung mmap. Dirty page cua mmap
        nam trong RSS cua process: da do ghi 200 MB qua memmap -> RSS +200 MB,
        va flush() (msync) KHONG lam giam, phai madvise(MADV_DONTNEED) moi
        tra ve. Buffered write day dirty page vao page cache cua kernel,
        khong tinh vao RSS -> so do duoc la RAM cua rieng GFP.

    Delta van bao gom vai MB COW overhead cua Python (refcount cham vao trang
    ke thua tu cha) -- khong dang ke so voi graph cua GFP.
    """
    import resource
    from snapml import GraphFeaturePreprocessor
    try:
        try:
            with open('/proc/self/clear_refs', 'w') as f:
                f.write('5')
        except OSError:
            pass
        rss0 = _rss_mb()

        g = GraphFeaturePreprocessor()
        g.set_params(cfg)
        n_cols = None
        t0 = time.time()
        with open(out_path, 'wb', buffering=1024 * 1024) as fo:
            for i in range(0, len(X), B):
                o = g.transform(X[i:i + B])
                if n_cols is None:
                    n_cols = o.shape[1]
                elif o.shape[1] != n_cols:
                    raise RuntimeError(f'so cot GFP doi giua cac lo: '
                                       f'{n_cols} -> {o.shape[1]}')
                fo.write(np.ascontiguousarray(o, dtype=np.float32).tobytes())
        dt = time.time() - t0

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 - rss0
        q.put(('ok', n_cols, dt, peak))
    except Exception as e:
        q.put(('err', repr(e), 0.0, 0.0))


class _InlineQueue:
    def __init__(self):
        self._v = None

    def put(self, v):
        self._v = v

    def get(self, *a, **kw):
        return self._v


def run_gfp(cfg, B, out_path, poll=30.0):
    """Tra ve (n_cols, thoi_gian_giay, delta_peak_RSS_MB cua rieng GFP).

    Doi ket qua bang vong poll thay vi q.get() chan vinh vien: neu con bi
    OOM-kill (rat de xay ra o B=1 full hoac o tw_control) thi cha phat hien
    qua p.is_alive() va bao loi kem exitcode, thay vi treo.
    """
    try:
        ctx = mp.get_context('fork')
    except ValueError:                          # khong co fork -> khong do duoc RAM
        q = _InlineQueue()
        _gfp_worker(cfg, X, B, out_path, q)
        status, a, dt, _ = q.get()
        if status != 'ok':
            raise RuntimeError(f'GFP worker loi: {a}')
        return a, dt, float('nan')

    q = ctx.Queue()
    p = ctx.Process(target=_gfp_worker, args=(cfg, X, B, out_path, q))
    p.start()
    msg = None
    while msg is None:
        try:
            msg = q.get(timeout=poll)
        except queue.Empty:
            if not p.is_alive():
                try:                            # tranh race: con da put roi moi thoat
                    msg = q.get(timeout=5)
                except queue.Empty:
                    p.join()
                    raise RuntimeError(
                        f'GFP worker chet khong tra ket qua, exitcode='
                        f'{p.exitcode} (exitcode -9 = bi OOM-kill)')
    p.join()
    status, a, dt, peak = msg
    if status != 'ok':
        raise RuntimeError(f'GFP worker loi: {a}')
    print(f'    [RAM] GFP peak delta = {peak:.0f} MB (da tru RSS ke thua)')
    return a, dt, peak


# ----------------------------------------------------------------------------
# Dem lookahead THUC SU: giao dich k bi coi la co lookahead neu ton tai giao
# dich j > k TRONG CUNG LO, chung it nhat mot tai khoan (src hoac dst), va
# ts[j] - ts[k] <= tw. Can duoi 1-hop -- xem ghi chu dau file.
# ----------------------------------------------------------------------------
def lookahead_stats(ts, src, dst, B, tw):
    """Tra ve (lookahead_rate_%, span_mean, span_p50, span_p95) theo giay."""
    n = len(ts)
    if B <= 1:
        return 0.0, 0.0, 0.0, 0.0

    spans = []
    exposed = 0
    for start in range(0, n, B):
        end = min(start + B, n)
        w_ts = ts[start:end]
        spans.append(int(w_ts[-1] - w_ts[0]))   # stream da sort tang dan
        m = end - start
        if m < 2:
            continue
        w_src = src[start:end]
        w_dst = dst[start:end]

        pos = {}
        for k in range(m):
            pos.setdefault(int(w_src[k]), []).append(k)
            pos.setdefault(int(w_dst[k]), []).append(k)

        for k in range(m - 1):
            tk = w_ts[k]
            for acct in (int(w_src[k]), int(w_dst[k])):
                lst = pos[acct]
                i = bisect_right(lst, k)
                if i < len(lst) and (w_ts[lst[i]] - tk) <= tw:
                    exposed += 1
                    break

    spans = np.asarray(spans, dtype=np.float64)
    return (100.0 * exposed / n, float(spans.mean()),
            float(np.percentile(spans, 50)), float(np.percentile(spans, 95)))


# ----------------------------------------------------------------------------
def fit_eval(Ftr, Fva, Fte, seed, spw='auto'):
    ytr = y[tr]
    spw_v = ((ytr == 0).sum() / max((ytr == 1).sum(), 1)) if spw == 'auto' else float(spw)
    m = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8,
                      tree_method='hist', eval_metric='aucpr', n_jobs=4,
                      scale_pos_weight=spw_v,
                      early_stopping_rounds=30, random_state=seed)
    m.fit(Ftr, ytr, eval_set=[(Fva, y[va])], verbose=False)
    s_va, s_te = m.predict_proba(Fva)[:, 1], m.predict_proba(Fte)[:, 1]
    thr = find_best_threshold(y[va], s_va)
    res = evaluate(y[te], s_te, thr)
    res['best_iteration'] = int(getattr(m, 'best_iteration', -1))
    res['spw'] = spw_v
    imp = m.get_booster().get_score(importance_type='gain')
    res['top_gain'] = ';'.join(k for k, _ in
                               sorted(imp.items(), key=lambda kv: -kv[1])[:10])
    return res

def load_df(csv, N):
    """Doc CSV. Neu N is None -> doc het.

    Neu N la so: KHONG dung nrows. Split trong file la lien tuc theo thoi gian
    (train -> val -> test) nen nrows se cat ra 100% train, lam val/test rong
    va lam vo find_best_threshold + early_stopping. Thay vao do lay N dong dau
    cua MOI vung theo ti le. Thu tu thoi gian trong tung vung duoc giu nguyen,
    chi sinh khoang trong giua cac vung (GFP chiu duoc, chi tia bot).
    """
    if N is None:
        return pd.read_csv(csv, dtype=DT)

    sp = pd.read_csv(csv, usecols=['split'])['split'].to_numpy()
    keep = np.zeros(len(sp), bool)
    for name in ('train', 'val', 'test'):
        idx = np.where(sp == name)[0]
        if len(idx) == 0:
            raise ValueError(f'khong tim thay split="{name}" trong {csv}')
        k = min(max(int(round(N * len(idx) / len(sp))), 2), len(idx))
        keep[idx[:k]] = True
    skip = np.where(~keep)[0] + 1               # +1 vi dong 0 la header
    print(f'[smoke] lay mau theo ti le: {keep.sum():,}/{len(sp):,} dong '
          f'(co khoang trong thoi gian giua cac vung)')
    return pd.read_csv(csv, dtype=DT, skiprows=skip)


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='dataset_high/HI-Small_Trans_split_index.csv')
    ap.add_argument('--n', type=int, default=200_000,
                    help='tong so dong lay mau (theo ti le tren ca 3 split); '
                         '0 = full. Mac dinh 200k de smoke test.')
    ap.add_argument('--batches', default='1,128,2048')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--tw-control', action='store_true',
                    help='chay them bien the time_window rat lon o B dau va B cuoi')
    ap.add_argument('--spw', default='auto',
                    help='auto = neg/pos (~1325); hoac so cu the. Paper tune trong (1,10).')
    ap.add_argument('--out', default='lookahead_batch_curve.csv')
    ap.add_argument('--tmpdir', default='.',
                    help='noi ghi file .f32 tam chua output cua GFP')
    args = ap.parse_args()

    N = args.n if args.n > 0 else None
    BATCHES = [int(b) for b in args.batches.split(',')]
    SEEDS = [int(s) for s in args.seeds.split(',')]

    # --- doc du lieu -------------------------------------------------------
    df = load_df(args.csv, N)
    ts = (pd.to_datetime(df['Timestamp']).astype('int64') // 10 ** 9).to_numpy()
    assert (np.diff(ts) >= 0).all(), 'Stream chua sort theo thoi gian!'

    src = (df['From Bank'] + '_' + df['Account']).astype('category').cat.codes.to_numpy()
    dst = (df['To Bank'] + '_' + df['Account.1']).astype('category').cat.codes.to_numpy()

    cur = df['Receiving Currency'].to_numpy()
    missing = sorted(set(cur) - set(FX_USD))
    assert not missing, f'thieu ty gia cho: {missing}'
    amt = df['Amount Received'].to_numpy(dtype=np.float64) * \
          np.array([FX_USD[c] for c in cur])
    print(f'[fx] amount -> USD | median theo loai tien: '
          f'{pd.Series(amt).groupby(pd.Series(cur)).median().round(0).to_dict()}')

    X = np.column_stack([np.arange(len(df)), src, dst, ts, amt]).astype(np.float64)

    y = df['Is Laundering'].to_numpy()
    split = df['split'].to_numpy()
    tr, va, te = split == 'train', split == 'val', split == 'test'
    assert tr.sum() and va.sum() and te.sum(), \
        f'split rong: train={tr.sum()} val={va.sum()} test={te.sum()}'
    print(f'{len(df):,} tx | train {tr.sum():,} val {va.sum():,} test {te.sum():,} '
          f'| pos {y.sum():,} ({100 * y.mean():.3f}%)')
    _u = np.unique(ts)
    _res = int(np.diff(_u).min()) if len(_u) > 1 else 0
    print(f'span dữ liệu = {(ts[-1] - ts[0]) / 86400:.2f} ngày | '
          f'độ phân giải timestamp = {_res}s')

    # --- in mac dinh cua GFP de doi chieu voi Figure 6 cua paper -----------
    try:
        from snapml import GraphFeaturePreprocessor
        _defaults = GraphFeaturePreprocessor().get_params()
        print('\n[GFP defaults]')
        for k in sorted(_defaults):
            if 'bins' in k or k in ('lc-cycle_len', 'vertex_stats_cols'):
                print(f'  {k} = {_defaults[k]}')
        print()
    except Exception as e:
        print(f'[canh bao] khong doc duoc get_params(): {e!r}')

    # --- danh sach cac lan chay -------------------------------------------
    runs = [('main', B, CFG_BASE, TW) for B in BATCHES]
    if args.tw_control:
        for B in (BATCHES[0], BATCHES[-1]):
            runs.append(('tw_control', B, CFG_TW_LARGE, TW_LARGE))

    rows = []
    refs = {}          # variant -> F[te] cua B nho nhat trong variant do
    for variant, B, cfg, tw in runs:
        print(f'\n=== {variant}/B={B} (time_window={cfg["time_window"]}) ===')

        out_path = os.path.join(args.tmpdir, f'_gfp_{variant}_{B}.f32')
        n_cols, dt, peak_mb = run_gfp(cfg, B, out_path)
        F = np.memmap(out_path, dtype=np.float32, mode='r',
                      shape=(len(X), n_cols))[:, 5:]    # bo 5 cot raw cua X
        n_feat = F.shape[1]

        la, span_mean, span_p50, span_p95 = lookahead_stats(ts, src, dst, B, tw)
        print(f'    lookahead={la:.2f}% | span mean/p50/p95 = '
              f'{span_mean:.0f}/{span_p50:.0f}/{span_p95:.0f}s | '
              f'gfp={dt:.1f}s | feat={n_feat}')

        # doc tu dia MOT lan cho ca 3 seed
        Ftr = np.asarray(F[tr])
        Fva = np.asarray(F[va])
        Fte = np.asarray(F[te])     # fancy index tren memmap -> da la mang moi

        if variant not in refs:
            refs[variant] = Fte
            blur = 0.0
        else:
            # rtol=0, atol=0 -> dem so hang test co it nhat 1 feature khac BIT
            # so voi B nho nhat cung variant. equal_nan=True: NaN == NaN.
            diff = ~np.isclose(Fte, refs[variant], rtol=0.0, atol=0.0,
                               equal_nan=True)
            blur = 100.0 * diff.any(axis=1).mean()
        print(f'    blur (khac-bit vs B={BATCHES[0]} cung variant) = {blur:.2f}%')

        for seed in SEEDS:
            r = fit_eval(Ftr, Fva, Fte, seed, args.spw)
            rows.append(dict(variant=variant, spw=r['spw'], top_gain=r['top_gain'], B=B, time_window=cfg['time_window'],
                             seed=seed,
                             f1=r['f1_minority'], pr_auc=r['pr_auc'],
                             precision=r['precision'], recall=r['recall'],
                             recall_at_fpr1=r['recall@fpr1%'],
                             precision_at_1000=r['precision@1000'],
                             threshold=r['threshold'],
                             best_iteration=r['best_iteration'],
                             lookahead=la, span_mean=span_mean,
                             span_p50=span_p50, span_p95=span_p95,
                             blur=blur, n_feat=n_feat,
                             gfp_s=dt, gfp_peak_mb=peak_mb))
            print(f"    seed={seed} f1={r['f1_minority']:.4f} "
                  f"pr_auc={r['pr_auc']:.4f} P={r['precision']:.4f} "
                  f"R={r['recall']:.4f} best_iter={r['best_iteration']}")
            pd.DataFrame(rows).to_csv(args.out, index=False)   # ghi tang dan

        del F, Ftr, Fva, Fte
        try:
            os.remove(out_path)
        except OSError:
            pass

    # --- tong hop ----------------------------------------------------------
    res = pd.DataFrame(rows)
    agg = (res.groupby(['variant', 'B'])
              .agg(f1_mean=('f1', 'mean'), f1_std=('f1', 'std'),
                   pr_auc_mean=('pr_auc', 'mean'),
                   best_iter_mean=('best_iteration', 'mean'),
                   lookahead=('lookahead', 'first'),
                   span_mean=('span_mean', 'first'),
                   span_p50=('span_p50', 'first'),
                   span_p95=('span_p95', 'first'),
                   blur=('blur', 'first'), n_feat=('n_feat', 'first'),
                   gfp_s=('gfp_s', 'first'), gfp_peak_mb=('gfp_peak_mb', 'first')))
    agg['delta_f1'] = agg['f1_mean'] - agg.groupby('variant')['f1_mean'].transform('first')
    print('\n' + agg.to_string(float_format=lambda v: f'{v:.4f}'))
    res.to_csv(args.out, index=False)
    print(f'\n-> {args.out}')