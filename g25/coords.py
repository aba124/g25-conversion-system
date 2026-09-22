"""Coordinate formatting and distances"""
from __future__ import annotations

import numpy as np


def format_row(name, coords, fmt="%.6f"):
    return str(name) + "," + ",".join(fmt % float(v) for v in coords)


def population_means(groups, coords, min_n=1):
    """Average coordinates per population label"""
    groups = np.asarray(groups, dtype=object)
    coords = np.asarray(coords, dtype=np.float64)
    names, out, counts = [], [], []
    for g in sorted(set(groups.tolist())):
        m = groups == g
        if int(m.sum()) < min_n:
            continue
        names.append(g)
        out.append(coords[m].mean(axis=0))
        counts.append(int(m.sum()))
    return np.array(names, dtype=object), np.asarray(out), np.asarray(counts)


def distances(target, sheet):
    t = np.asarray(target, dtype=np.float64)
    return np.sqrt(((np.asarray(sheet, dtype=np.float64) - t) ** 2).sum(axis=1))


def closest(target, names, sheet, n=20):
    d = distances(target, sheet)
    return [(str(names[i]), float(d[i])) for i in np.argsort(d)[:n]]
