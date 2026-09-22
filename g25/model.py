"""Trained model container and datasheet IO"""
from __future__ import annotations

import json
import numpy as np

_EMPTY = np.array([], dtype=np.int64)

_FIELDS = ("snp_chrom", "snp_pos", "freq", "loadings", "sv", "eig", "varexp",
           "pc_sd", "basis_scores", "panel_snp_rows", "panel_ind_cols",
           "panel_calib_cols")


class G25Model:
    """SNP loadings standardization and coordinate scaling"""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def k(self):
        return self.loadings.shape[1]

    @property
    def n_snp(self):
        return self.loadings.shape[0]

    def scale_factors(self):
        """Per PC multiplier turning unscaled into scaled"""
        if self.scale_mode == "eigen":
            f = self.eig / self.eig[0]
        elif self.scale_mode == "sqrt":
            f = self.sv / self.sv[0]
        else:
            f = np.ones(self.k)
        return np.asarray(f, dtype=np.float64)

    def to_unscaled(self, scores):
        return np.asarray(scores, dtype=np.float64) / self.pc_sd * self.unscaled_scale

    def to_scaled(self, scores):
        return self.to_unscaled(scores) * self.scale_factors()

    def save(self, path):
        d = {f: getattr(self, f, _EMPTY) for f in _FIELDS}
        d.update(snp_id=self.snp_id.astype("U32"), snp_a1=self.snp_a1,
                 snp_a2=self.snp_a2,
                 basis_id=np.asarray(self.basis_id, dtype="U64"),
                 basis_group=np.asarray(self.basis_group, dtype="U96"),
                 meta=np.asarray([json.dumps(self.meta)], dtype=object))
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"][0]))
        kw = {f: (z[f] if f in z.files else _EMPTY) for f in _FIELDS}
        kw.update(snp_id=z["snp_id"].astype(object),
                  snp_a1=z["snp_a1"].astype("U1"),
                  snp_a2=z["snp_a2"].astype("U1"),
                  basis_id=z["basis_id"].astype(object),
                  basis_group=z["basis_group"].astype(object),
                  meta=meta, scale_mode=meta.get("scale_mode", "sqrt"),
                  unscaled_scale=float(meta.get("unscaled_scale", 0.04)),
                  panel_prefix=meta.get("panel_prefix"))
        return cls(**kw)

    def describe(self):
        m = self.meta
        ve = self.varexp * 100.0
        return "\n".join([
            "model      : %s" % m.get("name", "(unnamed)"),
            "built      : %s" % m.get("built", "?"),
            "panel      : %s" % m.get("panel_prefix", "?"),
            "PCs        : %d" % self.k,
            "model SNPs : %d" % self.n_snp,
            "PCA basis  : %d individuals, %d populations"
            % (len(self.basis_id), len(set(self.basis_group))),
            "scaling    : %s, unscaled_scale=%.4g"
            % (self.scale_mode, self.unscaled_scale),
            "variance   : " + "  ".join("PC%d %.2f%%" % (i + 1, ve[i])
                                        for i in range(min(6, self.k))) + " ...",
        ])


def load_datasheet(path):
    """Read name,PC1,...,PCk rows"""
    names, rows = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 3 or line.startswith("#"):
                continue
            try:
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            names.append(parts[0])
            rows.append(vals)
    return np.array(names, dtype=object), np.asarray(rows, dtype=np.float64)


def write_datasheet(path, names, coords, fmt="%.6f"):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for nm, row in zip(names, coords):
            fh.write(str(nm) + "," + ",".join(fmt % v for v in row) + "\n")
    return path
