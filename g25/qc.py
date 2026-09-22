"""SNP quality control"""
from __future__ import annotations

import sys
import numpy as np

# long range LD regions in hg19 from Price et al
LONG_RANGE_LD_HG19 = [
    (1, 48000000, 52000000), (2, 86000000, 100500000),
    (2, 134500000, 138000000), (2, 183000000, 190000000),
    (3, 47500000, 50000000), (3, 83500000, 87000000),
    (3, 89000000, 97500000), (5, 44500000, 50500000),
    (5, 98000000, 100500000), (5, 129000000, 132000000),
    (5, 135500000, 138500000), (6, 25000000, 35000000),
    (6, 57000000, 64000000), (6, 140000000, 142500000),
    (7, 55000000, 66000000), (8, 7000000, 13000000),
    (8, 43000000, 50000000), (8, 112000000, 115000000),
    (10, 37000000, 43000000), (11, 46000000, 57000000),
    (11, 87500000, 90500000), (12, 33000000, 40000000),
    (12, 109500000, 112000000), (17, 40000000, 45000000),
    (20, 32000000, 34500000),
]


def long_range_ld_mask(chrom, pos, regions=None):
    bad = np.zeros(len(chrom), dtype=bool)
    for c, s, e in (LONG_RANGE_LD_HG19 if regions is None else regions):
        bad |= (chrom == c) & (pos >= s) & (pos <= e)
    return bad


def ld_prune(get_rows, chrom, r2=0.4, window=200, step=50, verbose=True):
    """Greedy windowed r2 pruning returning a keep mask"""
    n_snp = len(chrom)
    keep = np.ones(n_snp, dtype=bool)
    starts = np.flatnonzero(np.r_[True, chrom[1:] != chrom[:-1]])
    for cs, ce in zip(starts, np.r_[starts[1:], n_snp]):
        s = cs
        while s < ce:
            e = min(s + window, ce)
            x = np.asarray(get_rows(s, e), dtype=np.float32)
            x = x - x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1)
            sd[sd == 0] = 1.0
            x /= sd[:, None]
            c2 = ((x @ x.T) / x.shape[1]) ** 2
            np.fill_diagonal(c2, 0.0)
            local = keep[s:e].copy()
            for i in range(e - s):
                if local[i]:
                    drop = np.flatnonzero((c2[i] > r2) & local)
                    local[drop[drop > i]] = False
            keep[s:e] = local
            s += step
        if verbose:
            sys.stdout.write("\r    LD pruning %5.1f%%  kept %d"
                             % (100.0 * ce / n_snp, int(keep[:ce].sum())))
            sys.stdout.flush()
    if verbose:
        sys.stdout.write("\r    LD pruning done, kept %d of %d SNPs\n"
                         % (int(keep.sum()), n_snp))
    return keep
