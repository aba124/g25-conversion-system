"""Reference panel access for packed GENO and TGENO files"""
from __future__ import annotations

import os
import re
import sys
import numpy as np

from .eigenstrat import read_snp, read_ind, read_anno, MISSING

_SHIFTS = (6, 4, 2, 0)


def _unpack(raw, n):
    """2 bits per genotype to int8 with -1 for missing"""
    out = np.empty((raw.shape[0], raw.shape[1] * 4), dtype=np.int8)
    for i, sh in enumerate(_SHIFTS):
        out[:, i::4] = (raw >> sh) & 3
    out = np.ascontiguousarray(out[:, :n])
    out[out == 3] = MISSING
    return out


class Panel:
    def __init__(self, prefix=None, snp=None, ind=None, geno=None, anno=None,
                 verbose=True):
        if prefix is not None:
            snp = snp or prefix + ".snp"
            ind = ind or prefix + ".ind"
            geno = geno or prefix + ".geno"
            if anno is None and os.path.exists(prefix + ".anno"):
                anno = prefix + ".anno"
        for p in (snp, ind, geno):
            if p is None or not os.path.exists(p):
                raise FileNotFoundError("panel file missing: %s" % p)

        self.snp = read_snp(snp)
        self.ind_id, self.ind_sex, self.ind_group = read_ind(ind)
        self.anno = read_anno(anno) if anno and os.path.exists(anno) else []
        self.n_snp_total = len(self.snp["id"])
        self.n_ind_total = len(self.ind_id)

        size = os.path.getsize(geno)
        with open(geno, "rb") as fh:
            head = fh.read(64)
        m = re.match(rb"(TGENO|GENO)\s+(\d+)\s+(\d+)", head)
        if not m:
            raise ValueError("%s is not a packed geno file" % geno)
        self.kind = m.group(1).decode()
        if int(m.group(2)) != self.n_ind_total or int(m.group(3)) != self.n_snp_total:
            raise ValueError("%s header does not match the snp and ind files" % geno)

        # TGENO is one record per individual GENO is one record per SNP
        if self.kind == "TGENO":
            self.rlen = max(48, (self.n_snp_total + 3) // 4)
            nrec = self.n_ind_total
        else:
            self.rlen = max(48, (self.n_ind_total + 3) // 4)
            nrec = self.n_snp_total
        # header is either a full record or a 48 byte stub
        for hdr in (48, self.rlen):
            if hdr + self.rlen * nrec == size:
                break
        else:
            raise ValueError("%s is truncated or corrupt" % geno)
        self._mm = np.memmap(geno, dtype=np.uint8, mode="r", offset=hdr,
                             shape=(nrec, self.rlen))
        if verbose:
            print("  panel: %s  %d SNPs x %d individuals  (%s, %.1f GB)"
                  % (os.path.basename(geno), self.n_snp_total,
                     self.n_ind_total, self.kind, size / 1e9))

    def view(self, snp_idx=None, ind_idx=None, cache=False, verbose=True):
        return PanelView(self, snp_idx, ind_idx, cache, verbose)


class PanelView:
    """A SNP subset by individual subset window on a Panel"""

    def __init__(self, panel, snp_idx=None, ind_idx=None, cache=False,
                 verbose=True):
        self.panel = panel
        self.snp_idx = (np.arange(panel.n_snp_total) if snp_idx is None
                        else np.asarray(snp_idx, dtype=np.int64))
        self.ind_idx = (np.arange(panel.n_ind_total) if ind_idx is None
                        else np.asarray(ind_idx, dtype=np.int64))
        self.n_snp = len(self.snp_idx)
        self.n_ind = len(self.ind_idx)
        self._all_snps = self.n_snp == panel.n_snp_total
        self._packed = None
        self._dense = None
        if cache:
            self.cache_packed(verbose)

    def cache_packed(self, verbose=True):
        """Hold this view raw records in RAM"""
        if self.panel.kind != "TGENO":
            return
        if verbose:
            print("  caching %d individual records in RAM (%.2f GB)"
                  % (self.n_ind, self.n_ind * self.panel.rlen / 1e9))
        self._packed = np.array(self.panel._mm[self.ind_idx], dtype=np.uint8)

    def cache_dense(self, verbose=True, chunk=256):
        """Unpack the whole view to an int8 n_snp by n_ind array"""
        if verbose:
            print("  unpacking genotype matrix into RAM (%.2f GB)"
                  % (self.n_snp * self.n_ind / 1e9))
        self._dense = np.empty((self.n_snp, self.n_ind), dtype=np.int8)
        for c0 in range(0, self.n_ind, chunk):
            c1 = min(c0 + chunk, self.n_ind)
            self._dense[:, c0:c1] = self._columns_raw(c0, c1)
            if verbose:
                sys.stdout.write("\r    %5.1f%%" % (100.0 * c1 / self.n_ind))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write("\r    done      \n")
        return self._dense

    def _columns_raw(self, c0, c1):
        p = self.panel
        if p.kind == "TGENO":
            raw = (self._packed[c0:c1] if self._packed is not None
                   else p._mm[self.ind_idx[c0:c1]])
            g = _unpack(raw, p.n_snp_total)
            if not self._all_snps:
                g = g[:, self.snp_idx]
            return np.ascontiguousarray(g.T)
        g = _unpack(p._mm[self.snp_idx], p.n_ind_total)
        return np.ascontiguousarray(g[:, self.ind_idx[c0:c1]])

    def columns(self, c0, c1):
        """int8 n_snp by chunk for a block of individuals"""
        if self._dense is not None:
            return self._dense[:, c0:c1]
        return self._columns_raw(c0, c1)

    def rows(self, s0, s1):
        """int8 chunk by n_ind for a block of SNPs"""
        if self._dense is not None:
            return self._dense[s0:s1]
        p = self.panel
        if p.kind == "GENO":
            g = _unpack(p._mm[self.snp_idx[s0:s1]], p.n_ind_total)
            return np.ascontiguousarray(g[:, self.ind_idx])
        # unpack only the bytes holding these SNPs
        idx = self.snp_idx[s0:s1]
        b0, b1 = int(idx.min()) // 4, int(idx.max()) // 4 + 1
        raw = (self._packed[:, b0:b1] if self._packed is not None
               else p._mm[self.ind_idx, b0:b1])
        g = _unpack(raw, (b1 - b0) * 4)
        return np.ascontiguousarray(g[:, idx - b0 * 4].T)
