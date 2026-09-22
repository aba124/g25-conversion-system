"""Decide which panel individuals define the axes and which get projected"""
from __future__ import annotations

import re
import numpy as np

# AADR marks problem samples in the group ID itself
BAD_GROUP_PATTERNS = [
    r"ignore", r"qcremove", r"\bdup\b", r"_dup", r"-?dup\d*$", r"relative",
    r"contam", r"_lowq", r"notgroup", r"_outlier",
    # discovery samples were used to pick the SNPs
    r"-discovery", r"_wga", r"-wga",
    r"\bfather\b", r"\bmother\b", r"\bson\b", r"\bdaughter\b",
    r"\bbrother\b", r"\bsister\b", r"\btwin\b",
]
_BAD_RE = re.compile("|".join(BAD_GROUP_PATTERNS), re.IGNORECASE)

# -o and -oPCA mark individuals the curators judged ancestry outliers
_OUTLIER_RE = re.compile(r"-o[A-Za-z]*$")

# non human outgroups dated 0 BP matched exactly not by substring
NONHUMAN_GROUPS = {
    "Chimp", "Chimpanzee", "Gorilla", "Macaque", "Orangutan", "Bonobo",
    "Gibbon", "Marmoset", "Baboon", "Ancestor", "hg19ref", "Href", "Ref",
}

_SPLIT_SUFFIX_RE = re.compile(r"(?:[_-]\d+){1,2}$")
_SUFFIXES = (".DG", ".SG", ".SDG", ".AG", ".HO", ".WGA")


def is_nonhuman(group):
    return str(group).strip() in NONHUMAN_GROUPS


def is_flagged_group(group):
    return (bool(_BAD_RE.search(group)) or bool(_OUTLIER_RE.search(group))
            or is_nonhuman(group))


def clean_group_name(group, known=None):
    """Strip technical suffixes that split one population in two"""
    g = group
    for suf in _SUFFIXES:
        if g.endswith(suf):
            g = g[: -len(suf)]
    if known is not None:
        base = _SPLIT_SUFFIX_RE.sub("", g)
        if base != g and base in known:
            g = base
    return g


def id_suffix(gid):
    m = re.search(r"\.(DG|SDG|SG|AG|HO|WGA)$", gid)
    return m.group(1) if m else ""


def curate(anno, ind_ids, ind_groups, ho_snp_total, min_call_rate_basis=0.90,
           min_snps_project=15000, max_per_pop=30, calib_frac=0.15,
           modern_max_bp=0.0, allow_pseudohaploid_basis=False, verbose=True):
    """Return basis_idx calib_idx project_idx and per individual meta"""
    by_id = {a["id"]: a for a in anno}
    n = len(ind_ids)
    group = np.array([str(g) for g in ind_groups], dtype=object)
    gid = np.array([str(i) for i in ind_ids], dtype=object)

    date_bp = np.full(n, np.nan)
    snps_ho = np.zeros(n)
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)
    assess = np.empty(n, dtype=object)
    family = np.empty(n, dtype=object)
    country = np.empty(n, dtype=object)
    for i in range(n):
        a = by_id.get(gid[i])
        if a is None:
            assess[i] = family[i] = country[i] = ""
            continue
        date_bp[i], snps_ho[i] = a["date_bp"], a["snps_ho"]
        lat[i], lon[i] = a["lat"], a["lon"]
        assess[i], family[i], country[i] = (a["assessment"], a["family"],
                                            a["country"])

    suffix = np.array([id_suffix(g) for g in gid], dtype=object)
    is_modern = np.nan_to_num(date_bp, nan=1e9) <= modern_max_bp
    flagged = np.array([is_flagged_group(g) for g in group], dtype=bool)
    # pseudo haploid calls are homozygous so they distort covariance
    pseudohap = suffix == "SG"
    good = np.array([str(a).lower().startswith(("pass", "provisional_pass",
                                                "merge_pass"))
                     or str(a).strip() == "" for a in assess], dtype=bool)
    call_rate = np.where(snps_ho > 0, snps_ho / float(ho_snp_total), np.nan)

    eligible = (is_modern & ~flagged & good
                & (np.nan_to_num(call_rate, nan=1.0) >= min_call_rate_basis))
    if not allow_pseudohaploid_basis:
        eligible &= ~pseudohap

    # keep one individual per declared family preferring the best covered
    fam_key = np.array([f if f and f.lower() not in ("n/a", "na", "", "..")
                        else "" for f in family], dtype=object)
    best = {}
    for i in np.flatnonzero(eligible):
        k = fam_key[i]
        if k and (k not in best or snps_ho[i] > snps_ho[best[k]]):
            best[k] = i
    keep_fam = set(best.values())
    for i in np.flatnonzero(eligible):
        if fam_key[i] and i not in keep_fam:
            eligible[i] = False

    # cap big populations and reserve a slice of each for calibration
    stripped = set(clean_group_name(g) for g in group)
    clean = np.array([clean_group_name(g, stripped) for g in group], dtype=object)
    order = np.argsort(-snps_ho, kind="stable")
    by_pop = {}
    for i in order:
        if eligible[i]:
            by_pop.setdefault(clean[i], []).append(int(i))

    basis, calib = [], []
    for members in by_pop.values():
        m = len(members)
        n_hold = 0
        if calib_frac > 0 and m >= 3:
            n_hold = min(max(1, int(round(calib_frac * m))), m - 1)
        take = members[: max(0, min(max_per_pop, m - n_hold))]
        basis.extend(take)
        calib.extend(members[len(take):][:n_hold])
    basis = np.array(sorted(basis), dtype=np.int64)
    calib = np.array(sorted(calib), dtype=np.int64)

    project = np.flatnonzero(
        (np.nan_to_num(snps_ho, nan=0.0) >= min_snps_project) & ~flagged & good
    ).astype(np.int64)

    meta = {"id": gid, "group": group, "clean_group": clean, "suffix": suffix,
            "date_bp": date_bp, "snps_ho": snps_ho, "call_rate": call_rate,
            "assessment": assess, "is_modern": is_modern,
            "pseudohap": pseudohap, "flagged": flagged, "lat": lat, "lon": lon,
            "country": country}

    if verbose:
        print("  basis:   %6d individuals, %4d populations "
              "(present-day, diploid, unrelated, QC-passing)"
              % (len(basis), len(set(clean[basis]))))
        print("  calib:   %6d individuals reserved for chip calibration, "
              "%d populations" % (len(calib), len(set(clean[calib]))))
        print("  project: %6d individuals (%d ancient, %d present-day)"
              % (len(project), int((~is_modern[project]).sum()),
                 int(is_modern[project].sum())))
    return basis, calib, project, meta
