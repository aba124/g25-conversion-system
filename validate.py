"""Measure conversion accuracy on held out reference individuals"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from g25 import chipio, curate as cm, harmonize as hz
from g25.coords import distances, population_means
from g25.model import G25Model
from g25.panel import Panel
from g25.pca import standardization, standardize
from g25.project import (RestrictedBasis, apply_calibration, fit_calibration,
                         lsq_per_individual)
from convert import load_or_build_calibset


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--chip", required=True,
                    help="raw file whose marker set defines the test")
    ap.add_argument("-n", "--n-holdout", type=int, default=400)
    ap.add_argument("--methods", default="dot,lsq,em",
                    help="comma separated from dot lsq em")
    ap.add_argument("--min-call", type=float, default=0.90)
    ap.add_argument("--calib-n", type=int, default=1500)
    ap.add_argument("--calib-min-call", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def pick_holdout(model, panel, n, exclude, min_call=0.90, seed=0):
    """Present-day well genotyped individuals outside the basis and calibset"""
    basis = set(int(i) for i in model.panel_ind_cols) | set(int(i) for i in exclude)
    anno = {a["id"]: a for a in panel.anno}
    cand = []
    for i in range(panel.n_ind_total):
        if i in basis:
            continue
        gid, grp = str(panel.ind_id[i]), str(panel.ind_group[i])
        a = anno.get(gid)
        if a is None or not a["date_bp"] <= 0 or gid.endswith(".SG"):
            continue
        if cm.is_flagged_group(grp):
            continue
        if a["snps_ho"] / float(panel.n_snp_total) < min_call:
            continue
        cand.append(i)
    cand = np.array(cand, dtype=np.int64)
    if len(cand) > n:
        cand = np.sort(np.random.default_rng(seed).permutation(cand)[:n])
    return cand


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    mpath = args.model if args.model.endswith(".npz") else args.model + ".npz"
    model = G25Model.load(mpath)
    panel = Panel(prefix=model.panel_prefix)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    rec, _ = chipio.read_chip(args.chip)
    agree, _ = hz.check_build(rec, model.snp_id, model.snp_chrom, model.snp_pos)
    idx = hz.harmonize(rec, model.snp_id, model.snp_chrom, model.snp_pos,
                       model.snp_a1, model.snp_a2,
                       use_positions=agree >= 0.5)["idx"]
    print("chip %s: %d of %d model SNPs usable (%.1f%%)"
          % (os.path.basename(args.chip), len(idx), model.n_snp,
             100.0 * len(idx) / model.n_snp))

    cal_cols, cal_true = load_or_build_calibset(model, mpath, panel,
                                                args.calib_n,
                                                args.calib_min_call)
    if cal_true is None:
        sys.exit("not enough held-out individuals to fit a calibration")
    print("calibration individuals: %d" % len(cal_cols))

    hold = pick_holdout(model, panel, args.n_holdout, cal_cols, args.min_call,
                        args.seed)
    groups = np.array([cm.clean_group_name(str(panel.ind_group[i])) for i in hold],
                      dtype=object)
    print("holdout: %d present-day individuals from %d populations, in neither "
          "the axes nor the calibration" % (len(hold), len(set(groups.tolist()))))
    if len(set(groups.tolist())) < 10:
        print("  WARNING: few holdout populations, lower --min-call")

    mean2p, inv_sd = standardization(model.freq)
    m2, isd = standardization(model.freq[idx])
    rb = RestrictedBasis(model.loadings[idx], model.sv)

    print("reading holdout genotypes ...")
    Gh = panel.view(model.panel_snp_rows, hold, verbose=False).cache_dense()
    Xh = standardize(Gh, mean2p, inv_sd)
    # even 1 percent missing shrinks a dot product so truth is solved per person
    truth = lsq_per_individual(model.loadings, model.sv, Xh, Gh != -1)
    Xh_sub = Xh[idx]
    del Gh, Xh

    print("reading calibration genotypes ...")
    Gb = panel.view(model.panel_snp_rows[idx], cal_cols,
                    verbose=False).cache_dense()
    Xb = standardize(Gb, m2, isd)
    del Gb

    # R2 uses the whole panel spread not the holdout spread
    global_var = model.basis_scores.var(axis=0, ddof=1)
    tn, tmean, _ = population_means(groups, model.to_scaled(truth))
    nn = []
    for i in range(len(tn)):
        d = distances(tmean[i], tmean)
        d[i] = np.inf
        if np.isfinite(d).any():
            nn.append(d.min())
    nnd = float(np.median(nn)) if nn else float("nan")

    print()
    print("=" * 74)
    print("%-6s %-11s %8s %8s %9s %9s %8s"
          % ("method", "calibrated", "meanR2", "minR2", "med.err", "p90.err",
             "err/nnd"))
    print("=" * 74)
    results, raw_est = {}, {}
    for meth in methods:
        P_basis = rb.dot(Xb) if meth == "dot" else rb.lsq(Xb)
        W, b = fit_calibration(P_basis, cal_true)
        P_hat = (rb.dot(Xh_sub) if meth == "dot" else
                 rb.em(Xh_sub) if meth == "em" else rb.lsq(Xh_sub))
        raw_est[meth] = P_hat
        for on in (False, True):
            est = apply_calibration(P_hat, W, b) if on else P_hat
            mse = ((est - truth) ** 2).mean(axis=0)
            r2 = 1.0 - mse / np.maximum(global_var, 1e-300)
            err = np.sqrt(((model.to_scaled(est)
                            - model.to_scaled(truth)) ** 2).sum(axis=1))
            results[(meth, on)] = (r2, err)
            print("%-6s %-11s %8.4f %8.4f %9.5f %9.5f %8.2f"
                  % (meth, "yes" if on else "no", r2.mean(), r2.min(),
                     np.median(err), np.percentile(err, 90),
                     np.median(err) / nnd))
    print("=" * 74)
    print("med.err is the distance from the converted coordinate to the true")
    print("one in scaled units. err/nnd divides that by the median distance")
    print("between neighbouring populations (%.5f). Below 1 means a sample" % nnd)
    print("lands nearer its own population than its population sits from its")
    print("nearest neighbour")

    if "lsq" in raw_est and "em" in raw_est:
        print()
        print("lsq vs em: largest disagreement %.3g"
              % float(np.abs(raw_est["lsq"] - raw_est["em"]).max()))
        print("  EM imputation in PCA space has the least squares projection as")
        print("  its exact fixed point so these are one estimator not two")
    if ("dot", False) in results and ("dot", True) in results:
        a = np.median(results[("dot", False)][1])
        c = np.median(results[("dot", True)][1])
        print()
        print("Calibration improves the plain dot product %.1fx (%.5f -> %.5f)."
              % (a / max(c, 1e-12), a, c))
        print("Once calibrated dot lsq and em differ only by a fixed linear map")
        print("that the calibration absorbs so they converge")
    print()
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
