"""Reference panel imputation of SNPs a chip does not carry"""
from __future__ import annotations

import sys
import time
import numpy as np

MISSING = -1


def local_knn_impute(dose, observed, donors, mean2p, window=400, k=20,
                     min_obs=25, power=8.0, verbose=False):
    """Fill absent SNPs from locally best matching donors"""
    n_snp, n_donor = donors.shape
    out = np.asarray(mean2p, dtype=np.float32).copy()
    obs_idx = np.flatnonzero(observed)
    d = np.asarray(dose, dtype=np.float32)
    out[obs_idx] = d[obs_idx] if len(d) == n_snp else d

    t0 = time.time()
    nwin = (n_snp + window - 1) // window
    for wi in range(nwin):
        w0, w1 = wi * window, min((wi + 1) * window, n_snp)
        obs_w = np.flatnonzero(observed[w0:w1])
        miss_w = np.flatnonzero(~observed[w0:w1])
        if len(miss_w) == 0 or len(obs_w) < min_obs:
            continue
        D = donors[w0:w1]
        Dobs = D[obs_w]
        t = out[w0:w1][obs_w][:, None]
        # identity by state over the window
        ibs = np.where(Dobs != MISSING,
                       1.0 - np.abs(Dobs.astype(np.float32) - t) * 0.5, np.nan)
        with np.errstate(invalid="ignore"):
            sim = np.nan_to_num(np.nanmean(ibs, axis=0), nan=0.0)
        kk = min(k, n_donor)
        top = np.argpartition(-sim, kk - 1)[:kk]
        s = sim[top]
        w = np.power(np.maximum(s - s.min() + 1e-6, 1e-6), power)
        w /= w.sum()

        Dm = D[miss_w][:, top]
        vm = Dm != MISSING
        ww = np.broadcast_to(w, Dm.shape) * vm
        denom = ww.sum(axis=1)
        num = (np.where(vm, Dm.astype(np.float32), 0.0) * ww).sum(axis=1)
        good = denom > 1e-9
        vals = out[w0:w1].copy()
        vals[miss_w[good]] = num[good] / denom[good]
        out[w0:w1] = vals

        if verbose and (wi % 200 == 0 or wi == nwin - 1):
            sys.stdout.write("\r    knn imputation %5.1f%%  (%.0fs)"
                             % (100.0 * (wi + 1) / nwin, time.time() - t0))
            sys.stdout.flush()
    if verbose:
        sys.stdout.write("\r    knn imputation done      (%.0fs)\n"
                         % (time.time() - t0))
    return out


def imputation_accuracy(truth, imputed, observed, mean2p):
    """Squared error skill at the hidden SNPs against the 2p baseline"""
    m = ~observed
    t = np.asarray(truth, dtype=np.float64)[m]
    i = np.asarray(imputed, dtype=np.float64)[m]
    b = np.asarray(mean2p, dtype=np.float64)[m]
    ok = np.isfinite(t) & (t >= 0)
    t, i, b = t[ok], i[ok], b[ok]
    mse_i = float(np.mean((t - i) ** 2))
    mse_b = float(np.mean((t - b) ** 2))
    r = (float(np.corrcoef(t, i)[0, 1]) if t.std() > 0 and i.std() > 0
         else float("nan"))
    return {"n": int(len(t)), "mse": mse_i, "mse_baseline": mse_b,
            "r2_vs_baseline": 1.0 - mse_i / mse_b if mse_b > 0 else float("nan"),
            "dosage_r": r}
