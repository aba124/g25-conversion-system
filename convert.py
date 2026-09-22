"""Convert a raw DNA file into G25 style coordinates"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from g25 import chipio, coords as coords_mod, curate, harmonize as hz, impute as imp
from g25.model import G25Model, load_datasheet
from g25.panel import Panel
from g25.pca import standardization, standardize
from g25.project import (RestrictedBasis, apply_calibration, cv_calibration_r2,
                         fit_calibration, lsq_per_individual)

METHODS = ("lsq", "em", "dot", "knn")

# bump when the calibration maths changes so old cached maps are not reused
CALIBRATION_VERSION = "3"


def log(msg=""):
    print(msg, flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="raw data file or VCF")
    ap.add_argument("--model", required=True, help="model prefix or npz path")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None, help="default is next to the input")
    ap.add_argument("--method", default="lsq", choices=METHODS,
                    help="lsq is the default, the others are not improvements")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--drop-ambiguous", action="store_true",
                    help="discard A/T and C/G sites")
    ap.add_argument("--polarity", default="auto", choices=("auto", "a1", "a2"))
    ap.add_argument("--closest", type=int, default=15,
                    help="nearest populations to print, 0 is none")
    ap.add_argument("--vcf-sample", default=None)
    ap.add_argument("--min-r2", type=float, default=0.3,
                    help="drop imputed sites below this R2")
    ap.add_argument("--hard-calls", action="store_true",
                    help="ignore DS dosages and use GT")
    ap.add_argument("--knn-donors", type=int, default=1500)
    ap.add_argument("--knn-k", type=int, default=20)
    ap.add_argument("--knn-window", type=int, default=400)
    ap.add_argument("--knn-calib-n", type=int, default=250)
    ap.add_argument("--calib-n", type=int, default=1500)
    ap.add_argument("--calib-min-call", type=float, default=0.90)
    ap.add_argument("--scale-mode", default=None,
                    choices=("eigen", "sqrt", "none"))
    ap.add_argument("--g25-space", action="store_true",
                    help="map into published G25 space using align_to_g25.py")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def select_calibration_individuals(model, panel, n_max=1500, min_call=0.90):
    """Reference individuals never used to build the axes"""
    reserved = getattr(model, "panel_calib_cols", None)
    if reserved is not None and len(reserved) >= 200:
        return np.asarray(reserved, dtype=np.int64)[:n_max]

    basis = set(int(i) for i in model.panel_ind_cols)
    anno = {a["id"]: a for a in panel.anno}
    cand = []
    for i in range(panel.n_ind_total):
        if i in basis:
            continue
        gid, grp = str(panel.ind_id[i]), str(panel.ind_group[i])
        a = anno.get(gid)
        if a is None or not a["date_bp"] <= 0 or gid.endswith(".SG"):
            continue
        if curate.is_flagged_group(grp):
            continue
        if a["snps_ho"] / float(panel.n_snp_total) < min_call:
            continue
        cand.append(i)
    cand = np.asarray(sorted(cand), dtype=np.int64)
    if len(cand) <= n_max:
        return cand

    # stratify by population so no ancestry is left out
    by_pop = {}
    for i in cand:
        by_pop.setdefault(curate.clean_group_name(str(panel.ind_group[i])),
                          []).append(int(i))
    pops = sorted(by_pop)
    out, cap = [], 1
    while len(out) < n_max and cap < 1000:
        out = [i for p in pops for i in by_pop[p][:cap]]
        cap += 1
    return np.asarray(sorted(out[:n_max]), dtype=np.int64)


def load_or_build_calibset(model, model_path, panel, n_max, min_call,
                           chunk=250, verbose=True):
    """True coordinates of the held out individuals cached per model"""
    cache = model_path[:-4] + ".calibset.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        if int(z["n_max"]) == n_max and float(z["min_call"]) == min_call:
            return z["cols"], z["true"]

    cols = select_calibration_individuals(model, panel, n_max, min_call)
    if len(cols) < 200:
        return cols, None
    if verbose:
        log("  computing true coordinates for %d held-out reference "
            "individuals (one-off, cached)" % len(cols))
    mean2p, inv_sd = standardization(model.freq)
    out = np.zeros((len(cols), model.k))
    t0 = time.time()
    for c0 in range(0, len(cols), chunk):
        c1 = min(c0 + chunk, len(cols))
        G = panel.view(model.panel_snp_rows, cols[c0:c1],
                       verbose=False).cache_dense(verbose=False)
        out[c0:c1] = lsq_per_individual(model.loadings, model.sv,
                                        standardize(G, mean2p, inv_sd), G != -1)
        del G
        if verbose:
            sys.stdout.write("\r    %5.1f%%  (%.0fs)"
                             % (100.0 * c1 / len(cols), time.time() - t0))
            sys.stdout.flush()
    if verbose:
        sys.stdout.write("\r    done            \n")
    np.savez_compressed(cache, cols=cols, true=out, n_max=n_max,
                        min_call=min_call)
    return cols, out


def calibration_key(model_path, idx, method, extra=""):
    h = hashlib.sha1()
    h.update(os.path.basename(model_path).encode())
    h.update(str(os.path.getmtime(model_path)).encode())
    h.update(np.asarray(idx, dtype=np.int64).tobytes())
    h.update(method.encode())
    h.update(extra.encode())
    h.update(CALIBRATION_VERSION.encode())
    return h.hexdigest()[:16]


def read_input(args, model):
    rec, fmt = chipio.read_chip(args.input, args.vcf_sample, args.min_r2,
                                not args.hard_calls)
    log("  detected format : %s" % fmt)
    log("  markers in file : %d  (%.1f%% called)"
        % (len(rec["rsid"]), 100 * chipio.called_fraction(rec)))
    if rec.get("n_dropped_lowq"):
        log("  imputation QC   : dropped %d sites below R2 %.2f"
            % (rec["n_dropped_lowq"], args.min_r2))

    agree, n_chk = hz.check_build(rec, model.snp_id, model.snp_chrom,
                                  model.snp_pos)
    use_pos = agree >= 0.5
    log("  build check     : %.1f%% of %d rsID matches agree on position -> %s"
        % (100 * agree, n_chk, "same build as the panel" if use_pos
           else "DIFFERENT build, matching on rsID only"))
    return rec, use_pos


def match_markers(args, model, rec, use_pos):
    h = hz.harmonize(rec, model.snp_id, model.snp_chrom, model.snp_pos,
                     model.snp_a1, model.snp_a2, use_positions=use_pos,
                     drop_ambiguous=args.drop_ambiguous)
    s = h["stats"]
    idx, dose = h["idx"], h["dose"].astype(np.float32)
    n_amb = s["ambiguous_kept"] + s["ambiguous_dropped"]
    log("  matched         : %d (%d by rsID, %d by position)"
        % (s["matched"], s["matched_by_rsid"], s["matched_by_position"]))
    log("  strand-flipped  : %d" % s["strand_flipped"])
    log("  dropped         : %d allele mismatch, %d ambiguous"
        % (s["dropped_allele_mismatch"], s["ambiguous_dropped"]))
    if n_amb == 0:
        log("  strand-ambiguous: none, this panel has no A/T or C/G sites")
    log("  USABLE SNPs     : %d of %d model SNPs (%.1f%%)"
        % (len(idx), model.n_snp, 100.0 * len(idx) / model.n_snp))
    if len(idx) < 10000:
        log("  WARNING: fewer than 10,000 usable SNPs, coordinates will be noisy")

    if "dose" in rec:
        frac = float(np.mean(np.isfinite(rec["dose"][h["src"]])))
        frac_dose = hz.dosage_of_allele1(rec, h).astype(np.float32)
        bad = ~np.isfinite(frac_dose)
        frac_dose[bad] = dose[bad]
        dose = frac_dose
        log("  dosages         : fractional imputed dosage for %.1f%% of markers"
            % (100 * frac))

    fr = model.freq[idx]
    sign, r_un, r_amb = hz.detect_polarity(dose, fr, h["ambiguous"])
    if args.polarity != "auto":
        sign = 1 if args.polarity == "a1" else -1
    if n_amb and np.isfinite(r_amb):
        log("  allele polarity : r = %+.3f (unambiguous), %+.3f (A/T,C/G) -> "
            "counting allele %s" % (r_un, r_amb, "1" if sign > 0 else "2"))
    else:
        log("  allele polarity : r = %+.3f -> counting allele %s"
            % (r_un, "1" if sign > 0 else "2"))
    if np.isfinite(r_amb) and np.isfinite(r_un) and r_amb < 0.35 * r_un:
        log("  WARNING: ambiguous sites look strand-flipped, try --drop-ambiguous")
    if sign < 0:
        dose = 2.0 - dose
    return h, idx, dose


def marker_weights(rec, h, quiet):
    """Imputation R2 as per marker reliability"""
    if "dose" not in rec or not np.isfinite(rec["r2"]).any():
        return None
    r2m = np.asarray(rec["r2"], dtype=np.float64)[h["src"]]
    w = np.where(np.isfinite(r2m), r2m, 1.0)
    w[~np.isfinite(np.asarray(rec["dose"])[h["src"]])] = 1.0
    if not quiet:
        log("  marker weights  : imputation R2 used as reliability "
            "(mean %.3f, %d markers at 1.0)"
            % (float(w.mean()), int((w >= 0.999).sum())))
    return w


def project(args, model, rb, x, idx, dose, panel_holder):
    if args.method == "knn":
        panel = Panel(prefix=model.panel_prefix, verbose=not args.quiet)
        panel_holder.append(panel)
        nd = min(args.knn_donors, len(model.panel_ind_cols))
        sel = np.linspace(0, len(model.panel_ind_cols) - 1, nd).astype(int)
        donors = panel.view(model.panel_snp_rows,
                            model.panel_ind_cols[sel],
                            verbose=False).cache_dense(verbose=not args.quiet)
        obs = np.zeros(model.n_snp, dtype=bool)
        obs[idx] = True
        m2p, isd = standardization(model.freq)
        d_full = np.zeros(model.n_snp, dtype=np.float32)
        d_full[idx] = dose
        ximp = imp.local_knn_impute(d_full, obs, donors, m2p,
                                    window=args.knn_window, k=args.knn_k,
                                    verbose=not args.quiet)
        xs = ((ximp - m2p) * isd).astype(np.float64)
        return (model.loadings.T.astype(np.float64) @ xs) / model.sv, donors
    if args.method == "em":
        return rb.em(x), None
    if args.method == "dot":
        return rb.dot(x), None
    return rb.lsq(x), None


def calibrate(model, model_path, idx, args, raw, cache_dir, panel=None,
              donors=None, weights=None):
    os.makedirs(cache_dir, exist_ok=True)
    extra = ("%d-%d-%d" % (args.knn_donors, args.knn_k, args.knn_window)
             if args.method == "knn" else "")
    if weights is not None:
        extra += "-w" + hashlib.sha1(np.round(weights, 4).tobytes()).hexdigest()[:12]
    key = calibration_key(model_path, idx, args.method, extra)
    cpath = os.path.join(cache_dir, key + ".npz")
    if os.path.exists(cpath):
        z = np.load(cpath)
        log("  reusing cached calibration for this marker set (%s)" % key)
        return (apply_calibration(raw[None, :], z["W"], z["b"])[0],
                (z["r2"], z["rmse"]))

    if panel is None:
        panel = Panel(prefix=model.panel_prefix, verbose=not args.quiet)
    cols, P_true = load_or_build_calibset(model, model_path, panel,
                                          args.calib_n, args.calib_min_call,
                                          verbose=not args.quiet)
    if P_true is None:
        log("  WARNING: only %d individuals were held out of the axes so the "
            "basis is used instead, which over-fits the high PCs" % len(cols))
        cols, P_true = model.panel_ind_cols, model.basis_scores

    if args.method == "knn":
        P_hat, keep = calibrate_knn(model, panel, idx, args, donors, cols)
        P_true = P_true[keep]
    else:
        G = panel.view(model.panel_snp_rows[idx], cols,
                       verbose=False).cache_dense(verbose=False)
        m2, isd = standardization(model.freq[idx])
        X = standardize(G, m2, isd)
        del G
        if weights is not None:
            # degrade the refs like the sample so the fit matches
            rng = np.random.default_rng(0)
            w = weights[:, None].astype(np.float32)
            X = w * X + rng.normal(
                0.0, np.sqrt(np.maximum(w * (1.0 - w), 0.0)), X.shape
            ).astype(np.float32)
            log("  reference individuals degraded to the same imputation quality")
        rb = RestrictedBasis(model.loadings[idx], model.sv, weights=weights)
        log("  fitting on %d held-out reference individuals through the same "
            "%d markers" % (X.shape[1], len(idx)))
        P_hat = rb.dot(X) if args.method == "dot" else rb.lsq(X)
        del X

    r2, rmse, _ = cv_calibration_r2(P_hat, P_true)
    W, b = fit_calibration(P_hat, P_true)
    gain = float(np.mean([np.linalg.norm(W[k])
                          for k in range(min(10, model.k), model.k)]))
    log("  calibration fitted and cached (%s); mean gain on PC11+ = %.3f"
        % (key, gain))
    np.savez_compressed(cpath, W=W, b=b, r2=r2, rmse=rmse)
    return apply_calibration(raw[None, :], W, b)[0], (r2, rmse)


def calibrate_knn(model, panel, idx, args, donors, cols):
    """kNN impute a subsample of the calibration individuals"""
    n = min(args.knn_calib_n, len(cols))
    used = np.linspace(0, len(cols) - 1, n).astype(int)
    G = panel.view(model.panel_snp_rows[idx], np.asarray(cols)[used],
                   verbose=False).cache_dense(verbose=False)
    m2p, isd = standardization(model.freq)
    obs = np.zeros(model.n_snp, dtype=bool)
    obs[idx] = True
    U = model.loadings.astype(np.float64)
    P = np.zeros((n, model.k))
    log("  kNN-imputing %d reference individuals" % n)
    for i in range(n):
        gi = G[:, i].astype(np.float32)
        gi[gi < 0] = np.nan
        ok = np.isfinite(gi)
        d_full = np.zeros(model.n_snp, dtype=np.float32)
        d_full[idx] = np.where(ok, gi, m2p[idx])
        o = obs.copy()
        o[idx[~ok]] = False
        ximp = imp.local_knn_impute(d_full, o, donors, m2p,
                                    window=args.knn_window, k=args.knn_k)
        P[i] = (U.T @ ((ximp - m2p) * isd)) / model.sv
        if i % 10 == 0:
            sys.stdout.write("\r    %d/%d" % (i + 1, n))
            sys.stdout.flush()
    sys.stdout.write("\r    done        \n")
    return P, used


def print_closest(args, model, prefix, scaled, g25map):
    if g25map is not None:
        d = str(g25map["g25_dir"])
        sheets = [("present-day, published G25",
                   os.path.join(d, "Global25_PCA_modern_pop_averages.txt")),
                  ("ancient, published G25",
                   os.path.join(d, "Global25_PCA_pop_averages.txt"))]
        factors = g25map["scale_factors"]
    else:
        sheets = [("present-day", prefix + ".populations.modern.unscaled.txt"),
                  ("all, including ancient", prefix + ".populations.unscaled.txt")]
        factors = model.scale_factors()
    found = False
    for label, sheet in sheets:
        if not os.path.exists(sheet):
            continue
        found = True
        names, mat = load_datasheet(sheet)
        log()
        log("Closest reference populations - %s (scaled distance):" % label)
        for i, (nm, dd) in enumerate(coords_mod.closest(scaled, names,
                                                        mat * factors,
                                                        args.closest)):
            log("   %2d. %-48s %.5f" % (i + 1, nm, dd))
    if not found:
        log("\n(no datasheet next to %s, run build_model.py first)" % prefix)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    model_path = args.model if args.model.endswith(".npz") else args.model + ".npz"
    if not os.path.exists(model_path):
        sys.exit("model not found: %s\nBuild one with build_model.py" % model_path)
    model = G25Model.load(model_path)
    if args.scale_mode:
        model.scale_mode = args.scale_mode
    prefix = model_path[:-4]
    name = args.name or os.path.splitext(os.path.basename(args.input))[0]
    out_prefix = args.out or os.path.splitext(os.path.abspath(args.input))[0]
    cache_dir = args.cache_dir or os.path.join(
        os.path.dirname(os.path.abspath(model_path)), ".calibration")

    log("=" * 72)
    log("G25-style coordinate conversion")
    log("=" * 72)
    log(model.describe())
    log()

    log("[1/5] reading %s" % args.input)
    rec, use_pos = read_input(args, model)

    log("[2/5] matching markers to the model")
    h, idx, dose = match_markers(args, model, rec, use_pos)

    log("[3/5] projecting with method '%s'" % args.method)
    weights = marker_weights(rec, h, args.quiet)
    rb = RestrictedBasis(model.loadings[idx], model.sv, weights=weights)
    log("  axis coverage   : %.1f%% of the loading weight is on usable SNPs"
        % (100 * rb.coverage.mean()))
    m2, isd = standardization(model.freq[idx])
    x = standardize(dose[:, None], m2, isd)[:, 0]
    holder = []
    raw, donors = project(args, model, rb, x, idx, dose, holder)

    r2 = None
    if args.no_calibrate:
        log("[4/5] calibration skipped")
        cal = raw
    else:
        log("[4/5] chip-specific calibration")
        cal, r2 = calibrate(model, model_path, idx, args, raw, cache_dir,
                            panel=holder[0] if holder else None,
                            donors=donors, weights=weights)

    log("[5/5] coordinates")
    unscaled = model.to_unscaled(cal)
    g25map = None
    if args.g25_space:
        mp = prefix + ".g25map.npz"
        if not os.path.exists(mp):
            sys.exit("no alignment at %s, run align_to_g25.py first" % mp)
        g25map = np.load(mp, allow_pickle=True)
        unscaled = unscaled @ g25map["A"] + g25map["c"]
        scaled = unscaled * g25map["scale_factors"]
        log("  mapped into the published G25 space using %d anchors "
            "(held-out median error %.5f)"
            % (int(g25map["n_anchors"]), float(g25map["cv_median"])))
        en = int(g25map["euro_n"]) if "euro_n" in g25map.files else 0
        if en:
            t1 = 100.0 * int(g25map["euro_top1"]) / en
            log("  map accuracy on held-out European populations: %.0f%% exact, "
                "%.0f%% right region"
                % (t1, 100.0 * int(g25map["euro_region"]) / en))
            if t1 < 35:
                log("  WARNING: this map does not resolve within-Europe "
                    "structure. A European sample lands near the centre of the")
                log("  published European cloud whatever it really is, so treat "
                    "the European ranking below as uninformative.")
    else:
        scaled = model.to_scaled(cal)

    row_s = coords_mod.format_row(name, scaled)
    row_u = coords_mod.format_row(name, unscaled)
    log()
    log("--- SCALED (use these for distances and mixture models) " + "-" * 15)
    log(row_s)
    log()
    log("--- UNSCALED " + "-" * 58)
    log(row_u)
    log()
    for suffix, row in ((".g25.scaled.txt", row_s), (".g25.unscaled.txt", row_u)):
        with open(out_prefix + suffix, "w", newline="\n") as fh:
            fh.write(row + "\n")
        log("written: %s%s" % (out_prefix, suffix))

    if r2 is not None:
        rr, _ = r2
        log()
        log("Accuracy of this marker set, cross-validated on the held-out "
            "reference individuals")
        log("(R^2 = the fraction of each axis this marker set reproduces):")
        for i in range(model.k):
            log("   PC%-2d  R2=%6.3f  %s"
                % (i + 1, rr[i], "#" * int(round(max(rr[i], 0) * 30))))
        log("   mean R2 over %d PCs: %.4f" % (model.k, float(np.mean(rr))))

    if args.closest:
        print_closest(args, model, prefix, scaled, g25map)
    log()
    log("done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
