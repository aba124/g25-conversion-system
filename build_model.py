"""Build a G25 style PCA model from a reference panel"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from g25 import curate as curate_mod
from g25 import qc
from g25.coords import population_means
from g25.model import G25Model, write_datasheet
from g25.panel import Panel
from g25.pca import randomized_pca, snp_stats, standardization, standardize
from g25.project import lsq_per_individual

VALID = list("ACGT")


def human(t):
    return "%dm%02ds" % (int(t) // 60, int(t) % 60)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True,
                    help="prefix of the snp ind geno and anno files")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("-k", "--pcs", type=int, default=25)
    ap.add_argument("--maf", type=float, default=0.01)
    ap.add_argument("--snp-call-rate", type=float, default=0.95)
    ap.add_argument("--max-per-pop", type=int, default=30)
    ap.add_argument("--min-basis-call-rate", type=float, default=0.90)
    ap.add_argument("--calib-holdout", type=float, default=0.15,
                    help="fraction of every population held out of the axes "
                         "to fit the chip calibration on")
    ap.add_argument("--min-snps-project", type=int, default=15000)
    ap.add_argument("--keep-long-range-ld", action="store_true")
    ap.add_argument("--ld-prune", type=float, default=0.0,
                    help="r2 threshold, 0 is off")
    ap.add_argument("--power-iterations", type=int, default=3)
    ap.add_argument("--oversample", type=int, default=35)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--scale-mode", default="sqrt",
                    choices=("eigen", "sqrt", "none"),
                    help="sqrt uses singular values and matches published G25")
    ap.add_argument("--unscaled-scale", type=float, default=0.04)
    ap.add_argument("--no-project", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-snps", type=int, default=0, help="test builds only")
    ap.add_argument("--max-project", type=int, default=0, help="test builds only")
    return ap.parse_args(argv)


def select_snps(panel, basis_idx, args):
    snp = panel.snp
    keep = ((snp["chrom"] >= 1) & (snp["chrom"] <= 22)
            & np.isin(snp["a1"], VALID) & np.isin(snp["a2"], VALID)
            & (snp["a1"] != snp["a2"]))
    print("  autosomal biallelic ACGT SNPs: %d" % keep.sum())
    if not args.keep_long_range_ld:
        keep &= ~qc.long_range_ld_mask(snp["chrom"], snp["pos"])
        print("  after dropping long-range LD regions: %d" % keep.sum())

    stage1 = np.flatnonzero(keep)
    view1 = panel.view(stage1, basis_idx, cache=True)
    freq1, call1 = snp_stats(view1, chunk=args.chunk)
    maf = np.minimum(freq1, 1.0 - freq1)
    ok = np.isfinite(freq1) & (maf >= args.maf) & (call1 >= args.snp_call_rate)
    print("  after MAF >= %.3g and call rate >= %.3g: %d"
          % (args.maf, args.snp_call_rate, int(ok.sum())))
    final, freq = stage1[ok], freq1[ok]

    if args.max_snps and args.max_snps < len(final):
        sel = np.linspace(0, len(final) - 1, args.max_snps).astype(np.int64)
        final, freq = final[sel], freq[sel]
        print("  subsampled to %d SNPs for a test build" % len(final))

    if args.ld_prune > 0:
        tmp = panel.view(final, basis_idx)
        tmp._packed = view1._packed
        sub = np.linspace(0, len(basis_idx) - 1,
                          min(1500, len(basis_idx))).astype(int)
        pk = qc.ld_prune(lambda s, e: tmp.rows(s, e)[:, sub].astype(np.float32),
                         panel.snp["chrom"][final], r2=args.ld_prune)
        final, freq = final[pk], freq[pk]
        print("  after LD pruning at r2 > %.2f: %d" % (args.ld_prune, len(final)))

    return final, freq, view1._packed


def project_panel(panel, model, project_idx, basis_idx, basis_scores,
                  snp_rows, freq, chunk=128):
    """Least squares projection of everyone outside the basis"""
    mean2p, inv_sd = standardization(freq)
    basis_pos = {int(b): j for j, b in enumerate(basis_idx)}
    out = np.zeros((len(project_idx), model.k), dtype=np.float64)
    view = panel.view(snp_rows, project_idx)
    t0 = time.time()
    for c0 in range(0, len(project_idx), chunk):
        c1 = min(c0 + chunk, len(project_idx))
        g = view.columns(c0, c1)
        x = standardize(g, mean2p, inv_sd)
        obs = g != -1
        todo = [j for j in range(c1 - c0)
                if int(project_idx[c0 + j]) not in basis_pos]
        for j in range(c1 - c0):
            src = int(project_idx[c0 + j])
            if src in basis_pos:
                out[c0 + j] = basis_scores[basis_pos[src]]
        if todo:
            sel = np.asarray(todo)
            out[c0 + sel] = lsq_per_individual(model.loadings, model.sv,
                                               x[:, sel], obs[:, sel])
        sys.stdout.write("\r    projecting %5.1f%%  (%s)"
                         % (100.0 * c1 / len(project_idx), human(time.time() - t0)))
        sys.stdout.flush()
    sys.stdout.write("\r    projecting done      (%s)\n" % human(time.time() - t0))
    return out


def write_sheets(out, names, groups, meta, project_idx, model, scores):
    unscaled = model.to_unscaled(scores)
    scaled = model.to_scaled(scores)
    write_datasheet(out + ".individuals.scaled.txt", names, scaled)
    write_datasheet(out + ".individuals.unscaled.txt", names, unscaled)

    pnames, pscaled, counts = population_means(groups, scaled)
    _, punscaled, _ = population_means(groups, unscaled)
    write_datasheet(out + ".populations.scaled.txt", pnames, pscaled)
    write_datasheet(out + ".populations.unscaled.txt", pnames, punscaled)

    mask = meta["is_modern"][project_idx]
    mnames, mscaled, _ = population_means(groups[mask], scaled[mask])
    _, munscaled, _ = population_means(groups[mask], unscaled[mask])
    write_datasheet(out + ".populations.modern.scaled.txt", mnames, mscaled)
    write_datasheet(out + ".populations.modern.unscaled.txt", mnames, munscaled)

    with open(out + ".populations.info.tsv", "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("population\tn\tmodern\tmean_date_bp\tcountry\n")
        for p, c in zip(pnames, counts):
            m = groups == p
            fh.write("%s\t%d\t%s\t%.0f\t%s\n"
                     % (p, c, bool(meta["is_modern"][project_idx][m][0]),
                        np.nanmean(meta["date_bp"][project_idx][m]),
                        str(meta["country"][project_idx][m][0])))
    return len(pnames), len(mnames)


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print("[1/5] reading reference panel")
    panel = Panel(prefix=args.panel)

    print("[2/5] curating individuals")
    if panel.anno:
        basis_idx, calib_idx, project_idx, meta = curate_mod.curate(
            panel.anno, panel.ind_id, panel.ind_group, float(panel.n_snp_total),
            min_call_rate_basis=args.min_basis_call_rate,
            min_snps_project=args.min_snps_project,
            max_per_pop=args.max_per_pop, calib_frac=args.calib_holdout)
    else:
        print("  no anno file so everyone is treated as present-day")
        n = panel.n_ind_total
        basis_idx = project_idx = np.arange(n)
        calib_idx = np.array([], dtype=np.int64)
        meta = {"clean_group": np.array([str(g) for g in panel.ind_group],
                                        dtype=object),
                "is_modern": np.ones(n, bool), "date_bp": np.zeros(n),
                "country": np.array([""] * n, dtype=object)}
    if len(basis_idx) < 100:
        sys.exit("only %d individuals qualify for the basis" % len(basis_idx))

    print("[3/5] selecting SNPs")
    final, freq, packed = select_snps(panel, basis_idx, args)
    view = panel.view(final, basis_idx)
    view._packed = packed

    print("[4/5] principal component analysis")
    res = randomized_pca(view, freq, k=args.pcs, oversample=args.oversample,
                         n_power=args.power_iterations, chunk=args.chunk,
                         seed=args.seed)
    pc_sd = res["scores"].std(axis=0, ddof=1)
    pc_sd[pc_sd == 0] = 1.0
    print("  variance explained: " + " ".join(
        "PC%d=%.3f%%" % (i + 1, 100 * res["varexp"][i])
        for i in range(min(8, args.pcs))))

    snp = panel.snp
    model = G25Model(
        snp_id=snp["id"][final], snp_chrom=snp["chrom"][final],
        snp_pos=snp["pos"][final], snp_a1=snp["a1"][final],
        snp_a2=snp["a2"][final], freq=freq, loadings=res["loadings"],
        sv=res["sv"], eig=res["eig"], varexp=res["varexp"], pc_sd=pc_sd,
        basis_scores=res["scores"],
        basis_id=np.array([str(x) for x in panel.ind_id[basis_idx]], dtype=object),
        basis_group=np.array([str(x) for x in meta["clean_group"][basis_idx]],
                             dtype=object),
        panel_snp_rows=final.astype(np.int64),
        panel_ind_cols=np.asarray(basis_idx, dtype=np.int64),
        panel_calib_cols=np.asarray(calib_idx, dtype=np.int64),
        scale_mode=args.scale_mode, unscaled_scale=args.unscaled_scale,
        panel_prefix=os.path.abspath(args.panel),
        meta={"name": args.name or os.path.basename(args.out),
              "built": datetime.datetime.now().isoformat(timespec="seconds"),
              "panel_prefix": os.path.abspath(args.panel),
              "panel_kind": panel.kind, "k": int(args.pcs), "maf": args.maf,
              "snp_call_rate": args.snp_call_rate,
              "max_per_pop": args.max_per_pop,
              "calib_holdout": args.calib_holdout,
              "ld_prune_r2": args.ld_prune,
              "long_range_ld_removed": not args.keep_long_range_ld,
              "scale_mode": args.scale_mode,
              "unscaled_scale": args.unscaled_scale,
              "power_iterations": args.power_iterations, "seed": args.seed})
    model.save(args.out + ".npz")
    print("  wrote %s.npz (%.1f MB)"
          % (args.out, os.path.getsize(args.out + ".npz") / 1e6))

    if args.no_project:
        print("done in %s" % human(time.time() - t_start))
        return

    print("[5/5] projecting the full panel into the finished space")
    if args.max_project and args.max_project < len(project_idx):
        project_idx = project_idx[np.linspace(
            0, len(project_idx) - 1, args.max_project).astype(int)]
        print("  limited to %d individuals for a test build" % len(project_idx))
    scores = project_panel(panel, model, project_idx, basis_idx,
                           res["scores"], final, freq, chunk=args.chunk)

    names = np.array(["%s:%s" % (meta["clean_group"][i], panel.ind_id[i])
                      for i in project_idx], dtype=object)
    npop, nmod = write_sheets(args.out, names, meta["clean_group"][project_idx],
                              meta, project_idx, model, scores)
    print("  %d individuals, %d populations (%d present-day)"
          % (len(names), npop, nmod))
    print(model.describe())
    print("done in %s" % human(time.time() - t_start))


if __name__ == "__main__":
    main()
