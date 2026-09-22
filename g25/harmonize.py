"""Match a raw data file to the panel SNP set and fix strand and polarity"""
from __future__ import annotations

import numpy as np

# A=0 C=1 G=2 T=3 so the complement is 3 - code
_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}


def _encode(arr):
    out = np.full(len(arr), -1, dtype=np.int8)
    for ch, v in _CODE.items():
        out[arr == ch] = v
    return out


def harmonize(rec, snp_id, snp_chrom, snp_pos, snp_a1, snp_a2,
              use_positions=True, drop_ambiguous=False):
    """Align rec to a panel SNP table"""
    by_rs = {}
    for i, s in enumerate(snp_id):
        if s.startswith("rs"):
            by_rs.setdefault(s, i)
    by_pos = {}
    if use_positions:
        for i in range(len(snp_chrom)):
            by_pos.setdefault((int(snp_chrom[i]), int(snp_pos[i])), i)

    c1, c2 = _encode(rec["g1"]), _encode(rec["g2"])
    called = (c1 >= 0) & (c2 >= 0)
    rsids, chroms, positions = rec["rsid"], rec["chrom"], rec["pos"]

    tgt, src, how, seen = [], [], [], set()
    for i in np.flatnonzero(called):
        j = by_rs.get(rsids[i])
        s = 0
        if j is None and use_positions:
            j = by_pos.get((int(chroms[i]), int(positions[i])))
            s = 1
        if j is None or j in seen:
            continue
        seen.add(j)
        tgt.append(j)
        src.append(i)
        how.append(s)
    if not tgt:
        raise ValueError("no markers in common with the reference panel")

    tgt = np.asarray(tgt, dtype=np.int64)
    src_i = np.asarray(src, dtype=np.int64)
    how = np.asarray(how, dtype=np.int8)

    m1, m2 = _encode(snp_a1[tgt]), _encode(snp_a2[tgt])
    o1, o2 = c1[src_i], c2[src_i]
    biallelic = (m1 >= 0) & (m2 >= 0) & (m1 != m2)
    direct = ((o1 == m1) | (o1 == m2)) & ((o2 == m1) | (o2 == m2))
    f1, f2 = 3 - o1, 3 - o2
    flip = ((f1 == m1) | (f1 == m2)) & ((f2 == m1) | (f2 == m2)) & ~direct
    # both orientations fit at A/T and C/G so the vendor forward strand is used
    ambiguous = biallelic & (m1 + m2 == 3)

    keep = biallelic & (direct | flip)
    if drop_ambiguous:
        keep &= ~ambiguous
    a1 = np.where(flip, f1, o1)
    a2 = np.where(flip, f2, o2)
    dose = (a1 == m1).astype(np.int8) + (a2 == m1).astype(np.int8)

    idx = tgt[keep]
    order = np.argsort(idx, kind="stable")
    stats = {
        "chip_markers": int(len(rsids)), "chip_called": int(called.sum()),
        "matched": int(len(tgt)), "matched_by_rsid": int((how == 0).sum()),
        "matched_by_position": int((how == 1).sum()),
        "dropped_allele_mismatch": int((biallelic & ~(direct | flip)).sum()),
        "strand_flipped": int((flip & keep).sum()),
        "ambiguous_kept": int((ambiguous & keep).sum()),
        "ambiguous_dropped": int((ambiguous & ~keep).sum()),
        "used": int(keep.sum()),
    }
    return {"idx": idx[order], "dose": dose[keep][order],
            "ambiguous": ambiguous[keep][order], "flipped": flip[keep][order],
            "src": src_i[keep][order], "m1": m1[keep][order], "stats": stats}


def dosage_of_allele1(rec, h):
    """Turn a VCF ALT dosage into a dosage of the panel allele 1"""
    alt = _encode(rec["alt"])[h["src"]]
    alt = np.where(h["flipped"], 3 - alt, alt)
    d = np.asarray(rec["dose"], dtype=np.float64)[h["src"]]
    return np.where(alt == h["m1"], d, 2.0 - d)


def check_build(rec, snp_id, snp_chrom, snp_pos, sample=20000):
    """Fraction of rsID matches whose positions also agree"""
    by_rs = {s: i for i, s in enumerate(snp_id) if s.startswith("rs")}
    agree = tot = 0
    step = max(1, len(rec["rsid"]) // sample)
    for i in range(0, len(rec["rsid"]), step):
        j = by_rs.get(rec["rsid"][i])
        if j is None:
            continue
        tot += 1
        agree += int(snp_pos[j]) == int(rec["pos"][i])
    return (agree / tot) if tot else 0.0, tot


def detect_polarity(dose, freq_a1, ambiguous=None):
    """Work out which allele the panel genotypes count"""
    def corr(mask):
        d, p = (dose, freq_a1) if mask is None else (dose[mask], freq_a1[mask])
        ok = np.isfinite(p) & (p > 0.01) & (p < 0.99)
        d, p = d[ok].astype(np.float64), p[ok].astype(np.float64)
        if len(d) < 1000 or d.std() == 0 or p.std() == 0:
            return float("nan")
        return float(np.corrcoef(d, p)[0, 1])

    r_un = corr(None if ambiguous is None else ~ambiguous)
    r_amb = corr(ambiguous) if ambiguous is not None else float("nan")
    ref = r_un if np.isfinite(r_un) else r_amb
    return (-1 if (np.isfinite(ref) and ref < 0) else 1), r_un, r_amb
