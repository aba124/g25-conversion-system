"""Projecting a sample into a finished PCA space"""
from __future__ import annotations

import numpy as np


def standardize(dose, mean2p, inv_sd, missing_mask=None):
    x = np.asarray(dose, dtype=np.float32) - mean2p.astype(np.float32)
    x *= inv_sd.astype(np.float32)
    if missing_mask is not None:
        x[missing_mask] = 0.0
    return x


class RestrictedBasis:
    """Loadings cut down to the markers a sample has"""

    def __init__(self, loadings_S, sv, ridge=1e-6, weights=None):
        self.U = np.ascontiguousarray(loadings_S, dtype=np.float32)
        self.sv = np.asarray(sv, dtype=np.float64)
        k = self.U.shape[1]
        Ud = self.U.astype(np.float64)
        if weights is None:
            self.w = None
            G = Ud.T @ Ud
        else:
            self.w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, 1.0)
            G = (Ud * self.w[:, None]).T @ Ud
        self.gram = G
        self.gram_inv = np.linalg.inv(G + ridge * np.trace(G) / k * np.eye(k))
        self.coverage = np.diag(G)

    def _solve(self, X, corrected):
        single = X.ndim == 1
        Xm = X[:, None] if single else X
        A = self.U.T.astype(np.float64) @ Xm.astype(np.float64)
        Y = self.gram_inv @ A if corrected else A
        S = (Y / self.sv[:, None]).T
        return S[0] if single else S

    def lsq(self, X):
        """Least squares over the observed markers"""
        return self._solve(X, True)

    def dot(self, X):
        """Plain dot product for comparison only"""
        return self._solve(X, False)

    def em(self, X, n_iter=200, tol=1e-10):
        """EM fill of absent SNPs which converges to lsq"""
        single = X.ndim == 1
        Xm = np.asarray(X, dtype=np.float64)
        Xm = Xm[:, None] if single else Xm
        U = self.U.astype(np.float64)
        b = U.T @ Xm
        I_minus_G = np.eye(self.U.shape[1]) - self.gram
        Y = b.copy()
        for _ in range(n_iter):
            Yn = b + I_minus_G @ Y
            if np.max(np.abs(Yn - Y)) < tol * max(1.0, np.max(np.abs(Y))):
                Y = Yn
                break
            Y = Yn
        S = (Y / self.sv[:, None]).T
        return S[0] if single else S


def lsq_per_individual(loadings, sv, X, observed, ridge=1e-6):
    """Least squares with a separate gram per individual"""
    U = np.ascontiguousarray(loadings, dtype=np.float32)
    k = U.shape[1]
    sv = np.asarray(sv, dtype=np.float64)
    out = np.zeros((X.shape[1], k), dtype=np.float64)
    base = np.trace(U.T.astype(np.float64) @ U.astype(np.float64)) / k
    eye = ridge * base * np.eye(k)
    for j in range(X.shape[1]):
        o = observed[:, j]
        Uo = U[o].astype(np.float64)
        out[j] = np.linalg.solve(Uo.T @ Uo + eye,
                                 Uo.T @ X[o, j].astype(np.float64)) / sv
    return out


def fit_calibration(P_hat, P_true, ridge=1e-3):
    """Map restricted scores to full panel scores"""
    P_hat = np.asarray(P_hat, dtype=np.float64)
    P_true = np.asarray(P_true, dtype=np.float64)
    k = P_hat.shape[1]
    mu_h, mu_t = P_hat.mean(axis=0), P_true.mean(axis=0)
    Ph, Pt = P_hat - mu_h, P_true - mu_t
    G = Ph.T @ Ph
    W = np.linalg.solve(G + ridge * (np.trace(G) / k) * np.eye(k), Ph.T @ Pt)
    return W, mu_t - mu_h @ W


def apply_calibration(P, W, b):
    return np.asarray(P, dtype=np.float64) @ W + b


def cv_calibration_r2(P_hat, P_true, ridge=1e-3, folds=5, seed=0):
    """Cross validated R2 per PC"""
    P_hat = np.asarray(P_hat, dtype=np.float64)
    P_true = np.asarray(P_true, dtype=np.float64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(P_hat))
    pred = np.empty_like(P_true)
    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te)
        W, b = fit_calibration(P_hat[tr], P_true[tr], ridge)
        pred[te] = apply_calibration(P_hat[te], W, b)
    ss_res = ((pred - P_true) ** 2).sum(axis=0)
    ss_tot = ((P_true - P_true.mean(axis=0)) ** 2).sum(axis=0)
    rmse = np.sqrt(((pred - P_true) ** 2).mean(axis=0))
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-300), rmse, pred
