"""Convert every raw DNA file in a folder and build one combined datasheet"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

EXTS = (".txt", ".csv", ".tsv", ".vcf", ".gz")
SKIP = ("readme", "requirements", "run")


def find_inputs(folder):
    out = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        low = name.lower()
        if not os.path.isfile(p) or not low.endswith(EXTS):
            continue
        if low.startswith(SKIP) or ".g25." in low:
            continue
        out.append(p)
    return out


def read_row(path):
    with open(path, encoding="utf-8") as fh:
        return fh.readline().strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="samples")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None, help="default is <dir>/combined")
    ap.add_argument("--g25-space", action="store_true")
    ap.add_argument("--min-r2", type=float, default=0.3)
    args = ap.parse_args(argv)

    if not os.path.isdir(args.dir):
        sys.exit("no such folder: %s" % args.dir)
    inputs = find_inputs(args.dir)
    if not inputs:
        sys.exit("no raw data files found in %s" % args.dir)
    out = args.out or os.path.join(args.dir, "combined")
    print("found %d files in %s\n" % (len(inputs), args.dir))

    rows_s, rows_u, failed = [], [], []
    for i, path in enumerate(inputs, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        name = name[:-4] if name.lower().endswith(".vcf") else name
        cmd = [sys.executable, os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "convert.py"), path,
            "--model", args.model, "--name", name, "--closest", "0",
            "--quiet", "--min-r2", str(args.min_r2)]
        if args.g25_space:
            cmd.append("--g25-space")
        r = subprocess.run(cmd, capture_output=True, text=True)
        base = os.path.splitext(path)[0]
        if base.lower().endswith(".vcf"):
            base = base[:-4]
        scaled = base + ".g25.scaled.txt"
        if r.returncode != 0 or not os.path.exists(scaled):
            tail = (r.stderr or r.stdout).strip().splitlines()
            failed.append((name, tail[-1] if tail else "unknown error"))
            print("%2d/%d  %-28s FAILED" % (i, len(inputs), name))
            continue
        usable = ""
        for line in r.stdout.splitlines():
            if "USABLE SNPs" in line:
                usable = line.split(":")[1].strip()
        rows_s.append(read_row(scaled))
        rows_u.append(read_row(base + ".g25.unscaled.txt"))
        print("%2d/%d  %-28s %s" % (i, len(inputs), name, usable))

    for suffix, rows in ((".scaled.txt", rows_s), (".unscaled.txt", rows_u)):
        if rows:
            with open(out + suffix, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("\n".join(rows) + "\n")
            print("\nwrote %s%s (%d samples)" % (out, suffix, len(rows)))

    if failed:
        print("\n%d failed:" % len(failed))
        for name, why in failed:
            print("   %-28s %s" % (name, why[:70]))
    return 1 if failed and not rows_s else 0


if __name__ == "__main__":
    sys.exit(main())
