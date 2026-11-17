"""Score every baseline output with the selector's evaluator on the same test rows as VISOR.

The default point forecast of a method is the per-step median of all its output paths (one path for point models).
It is mapped to the path unit of the paper (divided by sigma * sqrt(h) of the row) and scored with MAE, MSE, endpoint
error and volatility error (visor.metrics). Inputs: px_tsfm_out/<prefix>_<name>.npz written by px_tsfm.py,
classic_ts.py, mm_ts.py, tats_ts.py, cfa_ts.py, with prefix fxt (FNSPID), ftxt / ftxfxt (FinTexTS) and tmfxt
(Time-MMD). Writes results/baselines_point.json {dataset: {name: metrics}}.
`python baselines/score_baselines.py`
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visor.metrics import features  # noqa: E402

TMMD = ["Algriculture", "Climate", "Economy", "Energy", "Public_Health", "Security", "SocialGood", "Traffic"]


def score(paths, sig, y):
    H = paths.shape[-1]
    u = sig.astype(np.float64)[:, None] * np.sqrt(np.arange(1, H + 1))[None]
    p = (np.median(paths.astype(np.float64), 1) / u).astype(np.float32)
    p3 = np.repeat(p[..., None], 3, -1)
    y = y.astype(np.float32)
    return {"mae": float(np.abs(p - y[..., 2]).mean()), "mse": float(((p - y[..., 2]) ** 2).mean()),
            "end": float(np.abs(p[:, -1] - y[:, -1, 2]).mean()),
            "volerr": float(np.abs(features(p3)[:, 0] - features(y)[:, 0]).mean())}


def outputs(prefixes):
    out = {}
    for pre in prefixes:
        for f in sorted(glob.glob(f"px_tsfm_out/{pre}_*.npz")):
            name = re.match(rf"px_tsfm_out/{pre}_(.+)\.npz", f).group(1)
            out.setdefault(name, f)
    return out


def main():
    res = {}
    fs = sorted(glob.glob("sel_dump/fsel_fnspid_te_g17_p*.npz"))
    if fs:
        D = [np.load(f, allow_pickle=True) for f in fs]
        rows = np.load("px_rows/fnspid_te23.npz")["rows"]
        sig, y = (np.concatenate([d[k] for d in D])[rows] for k in ("sig", "y"))
        res["FNSPID"] = {n: score(np.load(f)["paths"], sig, y) for n, f in outputs(["fxt"]).items()}
    if os.path.exists("sel_dump/fintexts_te.npz"):
        e = np.load("sel_dump/fintexts_te.npz", allow_pickle=True)
        r = np.load("px_rows/fintexts_te.npz")["rows"]
        # VISOR on FinTexTS is scored on the selector's fixed 5,000-row subset (--test_rows 5000, rng 0).
        sub = np.sort(np.random.default_rng(0).choice(len(r), 5000, replace=False))
        r = r[sub]
        res["FinTexTS"] = {n: score(np.load(f)["paths"][sub], e["sig"][r], e["y"][r]) for n, f in outputs(["ftxfxt", "ftxt"]).items()}
    if os.path.exists("sel_dump/tmmd_te.npz"):
        tz = np.load("sel_dump/tmmd_te.npz", allow_pickle=True)
        outs = outputs(["tmfxt"])
        for d in TMMD:
            m = tz["code"] == d
            res[d] = {n: score(np.load(f)["paths"][m], tz["sig"][m], tz["y"][m]) for n, f in outs.items()}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/baselines_point.json", "w"), indent=1)
    for d, v in res.items():
        print(d, {t: (round(x["mae"], 3), round(x["volerr"], 3)) for t, x in v.items()})


if __name__ == "__main__":
    main()
