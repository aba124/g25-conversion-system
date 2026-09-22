"""Readers for consumer raw data files and single sample VCFs"""
from __future__ import annotations

import gzip
import io
import os
import re
import numpy as np

NOCALL = {"", "-", "--", "0", "00", "N", "NN", ".", "..", "?"}
BASES = set("ACGT")
_R2_RE = re.compile(r"(?:^|;)(?:DR2|R2|INFO|MACH_R2)=([0-9.eE+-]+)")
_CHROM = {"X": "23", "Y": "24", "XY": "25", "PAR": "25", "MT": "26", "M": "26"}


def _open_text(path):
    if str(path).lower().endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8",
                                errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _chrom_to_int(c):
    c = str(c).strip().strip('"').upper().replace("CHR", "")
    c = _CHROM.get(c, c)
    return int(c) if c.isdigit() else 0


def _split_genotype(tok):
    """Vendor genotype token to two alleles with 0 for no call"""
    tok = tok.strip().strip('"').upper()
    if tok in NOCALL:
        return "0", "0"
    if len(tok) == 1:
        a = tok if tok in BASES else "0"
        return a, a
    a = tok[0] if tok[0] in BASES else "0"
    b = tok[1] if tok[1] in BASES else "0"
    return ("0", "0") if "0" in (a, b) else (a, b)


def detect_format(path):
    with _open_text(path) as fh:
        head = fh.read(8192)
    low = head.lower()
    if head.lstrip().startswith("##fileformat=VCF") or "\n#CHROM\t" in head:
        return "vcf"
    for key, name in (("ancestrydna", "ancestry"), ("23andme", "23andme"),
                      ("myheritage", "myheritage"), ("ftdna", "ftdna"),
                      ("familytreedna", "ftdna"), ("livingdna", "23andme"),
                      ("living dna", "23andme"), ("tellmegen", "23andme")):
        if key in low:
            return name
    for line in head.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.lower().replace('"', "").startswith("rsid"):
            return "ancestry" if len(re.split(r"[\t,]", line)) >= 5 else "generic"
        break
    return "generic"


def _pack(rsid, chrom, pos, g1, g2):
    return {"rsid": np.array(rsid, dtype=object),
            "chrom": np.array(chrom, dtype=np.int16),
            "pos": np.array(pos, dtype=np.int64),
            "g1": np.array(g1, dtype="U1"), "g2": np.array(g2, dtype="U1")}


def _read_columnar(path, want_cols):
    """Delimited vendor file with 5 allele columns or 4 genotype columns"""
    rsid, chrom, pos, g1, g2 = [], [], [], [], []
    with _open_text(path) as fh:
        for line in fh:
            if not line or line[0] == "#" or not line.strip():
                continue
            t = [x.strip().strip('"') for x in re.split(r"[\t,]", line.rstrip("\r\n"))]
            if len(t) < 4 or t[0].lower() in ("rsid", "rs", "snp", "snpid"):
                continue
            a, b = _split_genotype(t[3] + t[4] if want_cols >= 5 and len(t) >= 5
                                   else t[3])
            rsid.append(t[0])
            chrom.append(_chrom_to_int(t[1]))
            pos.append(int(t[2]) if t[2].isdigit() else 0)
            g1.append(a)
            g2.append(b)
    return _pack(rsid, chrom, pos, g1, g2)


def read_vcf(path, sample=None, min_r2=0.0, use_dosage=True):
    """One sample from a VCF using DS dosage when present"""
    rsid, chrom, pos, g1, g2 = [], [], [], [], []
    ref_a, alt_a, dose, r2s = [], [], [], []
    col = None
    n_lowq = 0
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            t = line.rstrip("\r\n").split("\t")
            if line.startswith("#CHROM"):
                if len(t) < 10:
                    raise ValueError("%s has no genotype columns" % path)
                names = t[9:]
                if sample is None:
                    col = 9
                elif sample in names:
                    col = 9 + names.index(sample)
                else:
                    raise ValueError("sample %r not in %s" % (sample, path))
                continue
            if col is None or len(t) <= col:
                continue
            ref, alt = t[3].upper(), t[4].upper()
            if ref not in BASES or alt not in BASES:
                continue

            q = float("nan")
            mq = _R2_RE.search(t[7])
            if mq:
                try:
                    q = float(mq.group(1))
                except ValueError:
                    pass
            if min_r2 > 0 and q == q and q < min_r2:
                n_lowq += 1
                continue

            fmt = t[8].split(":")
            fields = t[col].split(":")
            fmap = {k: fields[i] for i, k in enumerate(fmt) if i < len(fields)}
            d = float("nan")
            if use_dosage and "DS" in fmap:
                try:
                    d = float(fmap["DS"])
                except ValueError:
                    pass

            alleles = []
            for h in fmap.get("GT", "./.").replace("|", "/").split("/"):
                alleles.append(ref if h == "0" else alt if h == "1" else "0")
            if len(alleles) == 1:
                alleles *= 2
            a, b = (("0", "0") if len(alleles) != 2 or "0" in alleles
                    else tuple(alleles))
            # dosage present but no hard call so give harmonize a ref alt pair
            if d == d and a == "0":
                a, b = ref, alt

            ref_a.append(ref)
            alt_a.append(alt)
            dose.append(d)
            r2s.append(q)
            rsid.append(t[2] if t[2] not in (".", "") else ".")
            chrom.append(_chrom_to_int(t[0]))
            pos.append(int(t[1]) if t[1].isdigit() else 0)
            g1.append(a)
            g2.append(b)

    rec = _pack(rsid, chrom, pos, g1, g2)
    d = np.array(dose, dtype=np.float64)
    if np.isfinite(d).any():
        rec["dose"] = d
        rec["ref"] = np.array(ref_a, dtype="U1")
        rec["alt"] = np.array(alt_a, dtype="U1")
    rec["r2"] = np.array(r2s, dtype=np.float64)
    rec["n_dropped_lowq"] = n_lowq
    return rec


def read_chip(path, sample=None, min_r2=0.0, use_dosage=True):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    fmt = detect_format(path)
    if fmt == "vcf":
        rec = read_vcf(path, sample, min_r2, use_dosage)
        if "dose" in rec:
            fmt = "vcf (imputed dosages)"
    else:
        rec = _read_columnar(path, 5 if fmt == "ancestry" else 4)
    if len(rec["rsid"]) == 0:
        raise ValueError("%s: no genotype rows parsed as %s" % (path, fmt))
    return rec, fmt


def called_fraction(rec):
    return float(np.mean((rec["g1"] != "0") & (rec["g2"] != "0")))
