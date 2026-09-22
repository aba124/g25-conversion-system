"""Readers for EIGENSTRAT snp ind and anno files"""
from __future__ import annotations

import csv
import numpy as np

MISSING = -1

_CHROM = {"X": "23", "Y": "24", "XY": "25", "MT": "90", "M": "90"}


def _chrom_to_int(c):
    c = str(c).upper().replace("CHR", "")
    c = _CHROM.get(c, c)
    return int(c) if c.isdigit() else 0


def read_snp(path):
    """id chrom gendist pos allele1 allele2"""
    ids, chroms, ppos, a1, a2 = [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if len(t) < 6:
                continue
            ids.append(t[0])
            chroms.append(t[1])
            ppos.append(t[3])
            a1.append(t[4])
            a2.append(t[5])
    return {
        "id": np.array(ids, dtype=object),
        "chrom": np.array([_chrom_to_int(c) for c in chroms], dtype=np.int16),
        "pos": np.array(ppos, dtype=np.int64),
        "a1": np.array([s.upper() for s in a1], dtype="U1"),
        "a2": np.array([s.upper() for s in a2], dtype="U1"),
    }


def read_ind(path):
    """id sex group"""
    ids, sexes, groups = [], [], []
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if len(t) < 3:
                continue
            ids.append(t[0])
            sexes.append(t[1])
            groups.append(t[2])
    return (np.array(ids, dtype=object), np.array(sexes, dtype=object),
            np.array(groups, dtype=object))


def _num(s):
    try:
        return float(str(s).strip().replace(",", ""))
    except Exception:
        return float("nan")


def read_anno(path):
    # headers are prose so columns are found by prefix
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        rdr = csv.reader(fh, delimiter="\t")
        header = next(rdr)
        rows = [r for r in rdr if len(r) >= 20]

    def find(pred, default):
        for i, h in enumerate(header):
            if pred(h.lower().lstrip('"')):
                return i
        return default

    idx = {
        "id": 0,
        "group": find(lambda h: h.startswith("group id"), 14),
        "date_bp": find(lambda h: h.startswith("date mean in bp"), 10),
        "lat": find(lambda h: h.startswith("latitude"), 17),
        "lon": find(lambda h: h.startswith("longitude"), 18),
        "country": find(lambda h: h.startswith("political entity"), 16),
        "snps_ho": find(lambda h: "snps hit" in h and "ho snpset" in h, 27),
        "family": find(lambda h: h.startswith("family relations"), 31),
        "assessment": find(lambda h: h.strip().rstrip('"') == "assessment", 47),
    }
    out = []
    for r in rows:
        def g(k):
            i = idx[k]
            return r[i].strip() if i is not None and i < len(r) else ""
        out.append({
            "id": g("id"), "group": g("group"), "country": g("country"),
            "assessment": g("assessment"), "family": g("family"),
            "date_bp": _num(g("date_bp")), "lat": _num(g("lat")),
            "lon": _num(g("lon")),
            "snps_ho": _num(g("snps_ho")) if g("snps_ho") else 0.0,
        })
    return out
