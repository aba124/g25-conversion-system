"""Download the AADR reference panel from Harvard Dataverse"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://dataverse.harvard.edu/api/access/datafile/%d"

# AADR v66.p1 doi:10.7910/DVN/FFIDCW
DATASETS = {
    "HO": {
        "desc": "Human Origins: 584,131 SNPs x 27,594 individuals (~4.0 GB)",
        "files": {"v66_HO.geno": (13994808, 4029634650),
                  "v66_HO.snp": (13994527, 36800253),
                  "v66_HO.ind": (13994526, 1176253),
                  "v66_HO.anno": (13994528, 15465906)},
    },
    "1240K": {
        "desc": "1240K: 1,233,013 SNPs x 23,089 individuals (~7.1 GB)",
        "files": {"v66_1240K.geno": (13994829, 7117276654),
                  "v66_1240K.snp": (13994514, 77679819),
                  "v66_1240K.ind": (13994513, 1017460),
                  "v66_1240K.anno": (13994515, 13443726)},
    },
}


def fetch(url, dest, expect_size=None, chunk=1 << 20, retries=5):
    if expect_size and os.path.exists(dest) and os.path.getsize(dest) == expect_size:
        print("  %-16s already complete" % os.path.basename(dest))
        return
    tmp = dest + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0

    for attempt in range(retries):
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", "bytes=%d-" % have)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:
                    have = 0
                mode = "ab" if have else "wb"
                total = expect_size or int(r.headers.get("Content-Length", 0)) + have
                t0 = time.time()
                with open(tmp, mode) as fh:
                    while True:
                        buf = r.read(chunk)
                        if not buf:
                            break
                        fh.write(buf)
                        have += len(buf)
                        sys.stdout.write(
                            "\r  %-16s %6.2f / %.2f GB  %5.1f MB/s"
                            % (os.path.basename(dest), have / 1e9,
                               (total or have) / 1e9,
                               have / 1e6 / max(time.time() - t0, 1e-6)))
                        sys.stdout.flush()
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            print("\n  retry %d/%d after %s" % (attempt + 1, retries, e))
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            time.sleep(3 * (attempt + 1))
    else:
        raise SystemExit("could not download %s" % url)

    sys.stdout.write("\n")
    if expect_size and os.path.getsize(tmp) != expect_size:
        raise SystemExit("%s got %d bytes expected %d, delete %s and retry"
                         % (dest, os.path.getsize(tmp), expect_size, tmp))
    os.replace(tmp, dest)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/aadr")
    ap.add_argument("--dataset", default="HO", choices=sorted(DATASETS))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for k, v in sorted(DATASETS.items()):
            print("%-8s %s" % (k, v["desc"]))
        return

    spec = DATASETS[args.dataset]
    os.makedirs(args.out, exist_ok=True)
    print("AADR v66.p1 - %s" % spec["desc"])
    print("Source: Harvard Dataverse, doi:10.7910/DVN/FFIDCW")
    print("Destination: %s\n" % os.path.abspath(args.out))
    # smallest first so a broken link shows up before the big download
    for name, (fid, size) in sorted(spec["files"].items(), key=lambda kv: kv[1][1]):
        fetch(BASE % fid, os.path.join(args.out, name), size)
    print("\nDone. Build the model with:")
    print("  py build_model.py --panel %s --out models/%s25"
          % (os.path.join(args.out, "v66_" + args.dataset), args.dataset.lower()))


if __name__ == "__main__":
    main()
