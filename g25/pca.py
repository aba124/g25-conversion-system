"""Streaming randomized SVD of the standardized genotype matrix"""
from __future__ import annotations

import sys
import time
import numpy as np

MISSING = -1


def standardize(g, mean2p, inv_sd):
    """int8 to float32 centred and scaled with missing set to 0"""
    x = g.astype(np.float32)
    miss = g == MISSING
    x -= mean2p[:, None].astype(np.float32)
    x *= inv_sd[:, None].astype(np.float32)
    if miss.any():
        x[miss] = 0.0
    return x


def standardization(freq):
    p = np.asarray(freq, dtype=np.float64)
    var = 2.0 * p * (1.0 - p)
    inv_sd = np.where(var > 1e-12, 1.0 / np.sqrt(np.maximum(var, 1e-12)), 0.0)
    return 2.0 * p, inv_sd


def _tick(tag, done, total, t0):
    sys.stdout.write("\r    %-8s %5.1f%%  (%.0fs)"
                     % (tag, 100.0 * done / total, time.time() - t0))
    sys.stdout.flush()


def snp_stats(view, chunk=128, verbose=True):
    """Allele frequency and call rate per SNP in one pass"""
    tot = np.zeros(view.n_snp, dtype=np.int64)
    cnt = np.zeros(view.n_snp, dtype=np.int64)
    t0 = time.time()
    for c0 in range(0, view.n_ind, chunk):
        c1 = min(c0 + chunk, view.n_ind)
        g = view.columns(c0, c1)
        obs = g != MISSING
        cnt += obs.sum(axis=1)
        tot += np.where(obs, g, 0).sum(axis=1, dtype=np.int64)
        if verbose:
            _tick("freq", c1, view.n_ind, t0)
    if verbose:
        sys.stdout.write("\r    %-8s done    (%.0fs)\n" % ("freq", time.time() - t0))
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = np.where(cnt > 0, tot / (2.0 * np.maximum(cnt, 1)), np.nan)
    return freq, cnt / float(view.n_ind)


class _Mult:
    """X @ M and X.T @ Y accumulated over blocks of individuals"""

    def __init__(self, view, mean2p, inv_sd, chunk, verbose):
        self.view, self.chunk, self.verbose = view, chunk, verbose
        self.mean2p, self.inv_sd = np.asarray(mean2p), np.asarray(inv_sd)
        self.passes = 0

    def _blocks(self, tag):
        t0 = time.time()
        for c0 in range(0, self.view.n_ind, self.chunk):
            c1 = min(c0 + self.chunk, self.view.n_ind)
            yield c0, c1, standardize(self.view.columns(c0, c1),
                                      self.mean2p, self.inv_sd)
            if self.verbose:
                _tick(tag, c1, self.view.n_ind, t0)
        if self.verbose:
            sys.stdout.write("\r    %-8s done    (%.0fs)\n" % (tag, time.time() - t0))
        self.passes += 1

    def X(self, M):
        M = np.ascontiguousarray(M, dtype=np.float32)
        out = np.zeros((self.view.n_snp, M.shape[1]), dtype=np.float32)
        for c0, c1, x in self._blocks("X@M"):
            out += x @ M[c0:c1]
        return out

    def XT(self, Y):
        Y = np.ascontiguousarray(Y, dtype=np.float32)
        out = np.empty((self.view.n_ind, Y.shape[1]), dtype=np.float32)
        for c0, c1, x in self._blocks("X.T@Y"):
            out[c0:c1] = x.T @ Y
        return out


def _qr(A):
    q, _ = np.linalg.qr(A)
    return np.ascontiguousarray(q, dtype=np.float32)


def randomized_pca(view, freq, k=25, oversample=35, n_power=3, chunk=128,
                   seed=0, verbose=True):
    """Top k SVD returning SNP loadings and basis scores"""
    mean2p, inv_sd = standardization(freq)
    L = min(k + oversample, view.n_ind)
    rng = np.random.default_rng(seed)
    mul = _Mult(view, mean2p, inv_sd, chunk, verbose)

    if verbose:
        print("  randomized SVD: %d SNPs x %d individuals, k=%d, sketch=%d, "
              "%d power iterations" % (view.n_snp, view.n_ind, k, L, n_power))

    Y = mul.X(rng.standard_normal((view.n_ind, L)).astype(np.float32))
    for _ in range(n_power):
        Y = mul.X(_qr(mul.XT(_qr(Y))))
    Q = _qr(Y)
    B = mul.XT(Q).T

    Ub, sv, Vt = np.linalg.svd(np.asarray(B, dtype=np.float64),
                               full_matrices=False)
    loadings = Q @ np.ascontiguousarray(Ub[:, :k], dtype=np.float32)
    scores = np.ascontiguousarray(Vt[:k].T, dtype=np.float64)
    sv = np.asarray(sv[:k], dtype=np.float64)

    # fix the sign so rebuilding gives the same orientation
    for j in range(k):
        if loadings[np.argmax(np.abs(loadings[:, j])), j] < 0:
            loadings[:, j] *= -1.0
            scores[:, j] *= -1.0

    if verbose:
        print("  %d streaming passes over the genotype matrix" % mul.passes)
    return {"loadings": loadings, "scores": scores, "sv": sv,
            "eig": sv ** 2 / float(view.n_snp),
            "varexp": sv ** 2 / (float(view.n_snp) * float(view.n_ind))}
