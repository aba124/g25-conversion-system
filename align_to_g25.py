"""Fit a map from a local model into the published Global25 space"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from g25.model import G25Model, load_datasheet

SUFFIXES = (".DG", ".SG", ".SDG", ".AG", ".HO", ".WGA")
_NUM = re.compile(r"(?:[_-]\d+){1,2}$")

MODERN = "Global25_PCA_modern_pop_averages.txt"
ANCIENT = "Global25_PCA_pop_averages.txt"
MODERN_SCALED = "Global25_PCA_modern_pop_averages_scaled.txt"
ANCIENT_SCALED = "Global25_PCA_pop_averages_scaled.txt"

_NW = ("english", "scottish", "irish", "welsh", "orcadian", "icelandic",
       "norwegian", "danish", "swedish", "dutch", "belgian", "german",
       "french", "swiss", "austrian", "finnish", "estonian", "latvian",
       "lithuanian")
_SE = ("serbian", "croatian", "bosnian", "montenegrin", "macedonian",
       "bulgarian", "romanian", "moldovan", "greek", "albanian", "gagauz",
       "hungarian", "slovak", "czech", "polish", "ukrainian", "belarusian",
       "russian", "slovenian")
_SW = ("spanish", "spaniard", "portuguese", "basque", "italian", "sicilian",
       "sardinian", "maltese", "corsican")


def norm_label(s, drop_numeric=True):
    s = str(s)
    changed = True
    while changed:
        changed = False
        for x in SUFFIXES:
            if s.endswith(x):
                s, changed = s[: -len(x)], True
    return (_NUM.sub("", s) if drop_numeric else s).lower()


def _region(name):
    s = str(name).lower()
    for tag, keys in (("NW", _NW), ("SE", _SE), ("SW", _SW)):
        if any(s.startswith(k) for k in keys):
            return tag
    return None


def load_official(g25_dir):
    """Published population averages moderns and ancients together"""
    out, is_modern = {}, {}
    for fn, modern in ((MODERN, True), (ANCIENT, False)):
        path = os.path.join(g25_dir, fn)
        if not os.path.exists(path):
            raise SystemExit("missing %s\nPoint --g25-dir at the datasheets"
                             % path)
        names, mat = load_datasheet(path)
        for n, v in zip(names, mat):
            out[str(n)] = v
            is_modern[str(n)] = modern
    return out, is_modern


def official_scale_factors(g25_dir):
    """Per axis scaled over unscaled read off the published sheets"""
    num = np.zeros(25)
    den = np.zeros(25)
    for fu, fs in ((MODERN, MODERN_SCALED), (ANCIENT, ANCIENT_SCALED)):
        pu, ps = os.path.join(g25_dir, fu), os.path.join(g25_dir, fs)
        if not (os.path.exists(pu) and os.path.exists(ps)):
            continue
        nu, U = load_datasheet(pu)
        ns, S = load_datasheet(ps)
        iu = {str(n): i for i, n in enumerate(nu)}
        for n, s in zip(ns, S):
            i = iu.get(str(n))
            if i is not None:
                num += U[i] * s
                den += U[i] * U[i]
    return None if den.min() <= 0 else num / den


def build_anchors(model_prefix, official, drop_numeric=True):
    """Match local population averages to published ones by label"""
    names, mine = load_datasheet(model_prefix + ".populations.unscaled.txt")
    counts = {}
    info = model_prefix + ".populations.info.tsv"
    if os.path.exists(info):
        with open(info, encoding="utf-8") as fh:
            next(fh)
            for line in fh:
                t = line.rstrip("\n").split("\t")
                if len(t) >= 2 and t[1].isdigit():
                    counts[t[0]] = int(t[1])

    by_key = {}
    for k, v in official.items():
        by_key.setdefault(norm_label(k, drop_numeric), []).append(v)

    X, Y, lab, w = [], [], [], []
    for n, v in zip(names, mine):
        hits = by_key.get(norm_label(n, drop_numeric))
        if not hits:
            continue
        Y.append(np.mean(hits, axis=0))
        X.append(v)
        lab.append(str(n))
        w.append(float(counts.get(str(n), 1)))
    return (np.asarray(X), np.asarray(Y), np.asarray(lab, dtype=object),
            np.asarray(w))


def fit_map(X, Y, w=None, ridge=1e-3):
    """Ridge least squares Y ~ X A + c with per anchor weights"""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    w = np.ones(len(X)) if w is None else np.asarray(w, dtype=np.float64)
    w = w / w.mean()
    mx = (X * w[:, None]).sum(axis=0) / w.sum()
    my = (Y * w[:, None]).sum(axis=0) / w.sum()
    Xc, Yc = X - mx, Y - my
    G = (Xc * w[:, None]).T @ Xc
    k = X.shape[1]
    A = np.linalg.solve(G + ridge * (np.trace(G) / k) * np.eye(k),
                        (Xc * w[:, None]).T @ Yc)
    return A, my - mx @ A


def apply_map(P, A, c):
    return np.asarray(P, dtype=np.float64) @ A + c


def cv_score(X, Y, w, ridge, folds=8, seed=0):
    order = np.random.default_rng(seed).permutation(len(X))
    pred = np.empty_like(Y)
    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te)
        A, c = fit_map(X[tr], Y[tr], w[tr], ridge)
        pred[te] = apply_map(X[te], A, c)
    return pred


def region_accuracy(X, Y, lab, w, ridge, g25_dir):
    """Leave one out accuracy on European present-day anchors"""
    path = os.path.join(g25_dir, MODERN)
    if not os.path.exists(path):
        return 0, 0, 0
    onames, omat = load_datasheet(path)
    oset = set(str(n) for n in onames)
    idx = [i for i, l in enumerate(lab)
           if _region(l) is not None and str(l) in oset]
    if len(idx) < 8:
        return 0, 0, 0
    top1 = reg = 0
    for i in idx:
        tr = np.setdiff1d(np.arange(len(X)), [i])
        A, c = fit_map(X[tr], Y[tr], w[tr], ridge)
        p = apply_map(X[i:i + 1], A, c)[0]
        j = int(np.argmin(np.linalg.norm(omat - p, axis=1)))
        top1 += str(onames[j]) == str(lab[i])
        reg += _region(onames[j]) == _region(lab[i])
    return top1, reg, len(idx)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--g25-dir", required=True,
                    help="folder holding the published datasheets")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--no-numeric-merge", action="store_true")
    ap.add_argument("--out", default=None, help="default is <model>.g25map.npz")
    args = ap.parse_args(argv)

    prefix = args.model[:-4] if args.model.endswith(".npz") else args.model
    model = G25Model.load(prefix + ".npz")
    official, is_modern = load_official(args.g25_dir)
    print("published datasheets: %d populations" % len(official))

    X, Y, lab, w = build_anchors(prefix, official, not args.no_numeric_merge)
    print("anchors matched by label: %d" % len(X))
    if len(X) < 120:
        sys.exit("too few anchors (%d) to fit a 25x25 map" % len(X))
    print("   of which present-day: %d"
          % sum(1 for l in lab if is_modern.get(str(l), False)))

    best = None
    for ridge in (args.ridge, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 1.0):
        pred = cv_score(X, Y, w, ridge)
        err = np.linalg.norm(pred - Y, axis=1)
        if best is None or np.median(err) < best[1]:
            best = (ridge, float(np.median(err)), err, pred)
    ridge, med, err, pred = best
    print("\nbest ridge %.4g" % ridge)

    nn = []
    for i in range(0, len(Y), max(1, len(Y) // 400)):
        d = np.linalg.norm(Y - Y[i], axis=1)
        d[i] = np.inf
        nn.append(d.min())
    nnd = float(np.median(nn))
    Yc = Y - Y.mean(axis=0)
    r2 = 1.0 - ((pred - Y) ** 2).sum(axis=0) / np.maximum((Yc ** 2).sum(axis=0),
                                                          1e-300)
    print("held-out anchors (8-fold):")
    print("   median error                       %.5f" % med)
    print("   90th percentile                    %.5f" % np.percentile(err, 90))
    print("   median nearest-neighbour distance  %.5f" % nnd)
    print("   error / neighbour distance         %.2f" % (med / nnd))
    print("   mean R2 over 25 axes %.4f  (worst axis %.4f)"
          % (r2.mean(), r2.min()))

    top1, reg, n = region_accuracy(X, Y, lab, w, ridge, args.g25_dir)
    if n:
        print("\nheld-out European present-day anchors (%d):" % n)
        print("   nearest published population is the right one         : %.0f%%"
              % (100.0 * top1 / n))
        print("   nearest published population is the right broad region: %.0f%%"
              % (100.0 * reg / n))
        if top1 / max(n, 1) < 0.35:
            print("   WARNING: this map does not resolve within-Europe "
                  "structure. Mapped European samples land near the")
            print("   centre of the published European cloud whatever they "
                  "really are so a plausible answer there is not")
            print("   evidence of anything. Use it for continental placement only")

    A, c = fit_map(X, Y, w, ridge)
    fac = official_scale_factors(args.g25_dir)
    out = args.out or (prefix + ".g25map.npz")
    np.savez_compressed(out, A=A, c=c, ridge=ridge,
                        g25_dir=os.path.abspath(args.g25_dir),
                        scale_factors=(fac if fac is not None
                                       else model.scale_factors()),
                        n_anchors=len(X),
                        anchors=np.asarray(lab, dtype="U96"),
                        cv_median=med, cv_r2=r2, euro_top1=top1,
                        euro_region=reg, euro_n=n)
    print("\nwrote %s" % out)
    print("convert.py --g25-space will use it")


if __name__ == "__main__":
    main()
