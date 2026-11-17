"""Inputs of the Chart-Native image-space predictor (chart_native.py): per dataset, the history and the true future of every row
in the selector's own units, plus which rows of the selector files get image-generated candidates.

Units (same as the selector dumps): a row's history and future are its close (FNSPID: log price; FinTexTS / Time-MMD: the
z-scored series) minus the last input value, divided by the row's sig. So the future equals y[..., 2] * sqrt(h) and a
generated future f maps back to the selector's draws as f / sqrt(h).

Per dataset writes imgp_data/<ds>.npz with
  gtr_h, gtr_f      generator training windows (train split only; Time-MMD adds the file's other numeric columns, train part)
  gva_h, gva_f      validation windows for early stopping (at most 2000 val rows, fixed)
  {tr,va,te}_h, _f, _rows   rows that get candidates; _rows index the selector base file of that split
Row subsets: FNSPID train 30k / val 5k fixed random rows, test = px_rows/fnspid_te23 (the 10k rows the FNSPID selector is scored on);
FinTexTS train/val = px_rows/fintexts_{tr,va} (the rows the TSFM arm used),
test = the 5,000 rows the FinTexTS selector is scored on; Time-MMD = all rows.
`python -m visor.chart_native_data [ds ...]`
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

TMMD = ["Algriculture", "Climate", "Economy", "Energy", "Public_Health", "Security", "SocialGood", "Traffic"]
DS = ["fnspid", "fintexts"] + TMMD
from visor import paths

MEM = paths.FNSPID
S = 0.05


def base_files(ds, sp):
    """Selector base file(s) of a split, in the order selector.load concatenates them."""
    if ds == "fnspid":
        return sorted(glob.glob(f"sel_dump/fsel_fnspid_{sp}_g17_p*.npz"))
    if ds == "fintexts":
        return [f"sel_dump/fintexts_{sp}.npz"]
    return [f"sel_dump/tmmdC_{ds}_{sp}.npz"]


def load_base(ds, sp, keys=("day", "code", "sig", "y")):
    D = [np.load(f, allow_pickle=True) for f in base_files(ds, sp)]
    return {k: np.concatenate([d[k] for d in D]) for k in keys}


def fut_of(y):
    H = y.shape[1]
    return (y[:, :, 2].astype(np.float64) * np.sqrt(np.arange(1, H + 1))[None]).astype(np.float32)


def fnspid_hist(code, day, sig, L=90):
    """Last L closes up to and including day, as log(C / C_last) / sig (left-padded with the first value)."""
    out = np.zeros((len(code), L), np.float32)
    for c in np.unique(code):
        m = np.flatnonzero(code == c)
        g = pd.read_parquet(f"{MEM}/raw/{c}.parquet", columns=["datetime", "close"])
        t = g.datetime.values.astype("datetime64[D]")
        lc = np.log(g.close.to_numpy(np.float64))
        j = np.searchsorted(t, day[m].astype("datetime64[D]"), side="right")
        for r, jj in zip(m, j):
            h = lc[max(0, jj - L):jj]
            h = np.r_[np.full(L - len(h), h[0]), h]
            out[r] = (h - h[-1]) / sig[r]
    return out


def side_hist(side, code, day, sig_dump):
    """FinTexTS / Time-MMD history from the side file, in sig units; checks sig against the dump's (sig = S * std(diff))."""
    key = {(c, d): i for i, (c, d) in enumerate(zip(side["code"], side["day"]))}
    hist = side["hist"]
    hz = hist[[key[(c, str(d))] for c, d in zip(code, day)]].astype(np.float64)
    sig = np.maximum(np.diff(hz, axis=-1).std(-1), 1e-3)
    assert np.allclose(S * sig, sig_dump, rtol=1e-3, atol=1e-6), np.abs(S * sig - sig_dump).max()
    return ((hz - hz[:, -1:]) / sig[:, None]).astype(np.float32)


def build(ds):
    os.makedirs("imgp_data", exist_ok=True)
    out = {}
    B = {sp: load_base(ds, sp) for sp in ("tr", "va", "te")}
    if ds == "fnspid":
        H = {sp: fnspid_hist(B[sp]["code"], B[sp]["day"], B[sp]["sig"].astype(np.float64)) for sp in ("tr", "va")}
        te_rows = np.load("px_rows/fnspid_te23.npz")["rows"]
        sub = {sp: np.sort(np.random.default_rng(0).choice(len(B[sp]["day"]), n, replace=False)) for sp, n in (("tr", 30000), ("va", 5000))}
        sub["te"] = te_rows
        e = B["te"]
        H["te"] = np.zeros((len(e["day"]), 90), np.float32)
        H["te"][te_rows] = fnspid_hist(e["code"][te_rows], e["day"][te_rows], e["sig"][te_rows].astype(np.float64))
        out["gtr_h"], out["gtr_f"] = H["tr"], fut_of(B["tr"]["y"])
    elif ds == "fintexts":
        z = np.load("sel_dump/fintexts_side.npz", allow_pickle=True)
        side = {k: z[k] for k in ("code", "day", "hist")}
        H = {sp: side_hist(side, B[sp]["code"], B[sp]["day"], B[sp]["sig"]) for sp in ("tr", "va", "te")}
        sub = {sp: np.load(f"px_rows/fintexts_{sp}.npz")["rows"] for sp in ("tr", "va")}
        # Test: the 5,000 rows the FinTexTS selector is scored on (--test_rows 5000 picks them with default_rng(0) from all rows).
        sub["te"] = np.sort(np.random.default_rng(0).choice(len(B["te"]["day"]), 5000, replace=False))
        out["gtr_h"], out["gtr_f"] = H["tr"], fut_of(B["tr"]["y"])
    else:
        # History straight from flow_mlp's windows (same rows and order as the tmmdC dumps). The side file is keyed by
        # (series, last day), which is not unique for a few weekly windows, so it is not used here.
        from visor import flow_mlp as BT
        W = BT.windows()
        H = {}
        for sp in ("tr", "va", "te"):
            rows = [w for w in W[sp] if w[0] == ds]
            assert np.array_equal(np.array([w[1] for w in rows]), B[sp]["day"])
            h, f, sig = BT.norm(np.array([w[2] for w in rows]), np.array([w[3] for w in rows]))
            assert np.allclose(S * sig, B[sp]["sig"], rtol=1e-3, atol=1e-6)
            assert np.allclose(f, fut_of(B[sp]["y"]), atol=2e-2 * (1 + np.abs(f)))
            H[sp] = h.astype(np.float32)
        sub = {sp: np.arange(len(B[sp]["day"])) for sp in ("tr", "va", "te")}
        ex = [w for w in W["extra"] if w[0] == ds]
        gh, gf = H["tr"], fut_of(B["tr"]["y"])
        if ex:
            eh, ef, _ = BT.norm(np.array([w[2] for w in ex]), np.array([w[3] for w in ex]))
            gh, gf = np.r_[gh, eh.astype(np.float32)], np.r_[gf, np.clip(ef, -30, 30).astype(np.float32)]
        out["gtr_h"], out["gtr_f"] = gh, gf
        print(ds, "extra windows", len(ex), flush=True)
    vrow = np.sort(np.random.default_rng(1).choice(len(B["va"]["day"]), min(2000, len(B["va"]["day"])), replace=False))
    out["gva_h"], out["gva_f"] = H["va"][vrow], fut_of(B["va"]["y"])[vrow]
    for sp in ("tr", "va", "te"):
        r = sub[sp]
        out[f"{sp}_h"], out[f"{sp}_f"], out[f"{sp}_rows"] = H[sp][r], fut_of(B[sp]["y"])[r], r
    # Clip extreme normalised moves (as flow_mlp does for its flow) only in the generator's training windows.
    out["gtr_f"] = np.clip(out["gtr_f"], -30, 30)
    np.savez(f"imgp_data/{ds}.npz", **out)
    print(ds, {k: v.shape for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    os.makedirs("imgp_data", exist_ok=True)
    for d in (sys.argv[1:] or DS):
        build(d)
