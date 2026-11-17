"""Frozen-TSFM predictor design: the candidates of the selector files replaced by samples of a released TSFM.

FNSPID: the 10,000 rows of px_rows/fnspid_te23 (same rows the selector is scored on), Time-MMD: tmmdC_<ds>_te rows.
draws [N, K, H, 3] in the selector's volatility units: log return / (sig * sqrt(h)); every channel gets the close path.
`python -m visor.frozen_tsfm_candidates` (FNSPID test, Time-MMD test); `... fnspid <t>...` (FNSPID train/val);
`... fintexts <t>...`; `... <t>...` (Time-MMD train/val). <t> is a TSFM output of baselines/px_tsfm.py.
"""
import glob
import numpy as np

TSFM = ("chronos_t5b", "moirai11b", "toto")   # the released models that output sample paths, not quantiles
K = 32
DS = ["Algriculture", "Climate", "Economy", "Energy", "Public_Health", "Security", "SocialGood", "Traffic"]


def to_draws(paths, sig, H):
    u = sig[:, None, None] * np.sqrt(np.arange(1, H + 1))[None, None]
    d = (paths[:, :K].astype(np.float64) / u).astype(np.float32)
    return np.repeat(d[..., None], 3, -1)


def main():
    fs = sorted(glob.glob("sel_dump/fsel_fnspid_te_g17_p*.npz"))
    D = [np.load(f, allow_pickle=True) for f in fs]
    rows = np.load("px_rows/fnspid_te23.npz")["rows"]
    base = {k: np.concatenate([d[k] for d in D])[rows] for k in D[0].files}
    tz = np.load("sel_dump/tmmd_te.npz", allow_pickle=True)
    for t in TSFM:
        out = dict(base, draws=to_draws(np.load(f"px_tsfm_out/fxt_{t}.npz")["paths"], base["sig"].astype(np.float64), 10))
        np.savez(f"sel_dump/fsel_fnspid_te_g17_{t}.npz", **out)
        P = np.load(f"px_tsfm_out/tmfxt_{t}.npz")["paths"]
        for d in DS:
            e = dict(np.load(f"sel_dump/tmmdC_{d}_te.npz", allow_pickle=True)); m = tz["code"] == d
            assert np.array_equal(tz["day"][m], e["day"])
            e["draws"] = to_draws(P[m], e["sig"].astype(np.float64), 12)
            np.savez(f"sel_dump/tmmdC_{d}_te_{t}.npz", **e)
        print(t, out["draws"].shape, flush=True)


def trainval(t):
    """Same swap on the Time-MMD train and validation rows (stage 2: fine-tune the selector on TSFM candidates)."""
    for sp in ("tr", "va"):
        z = np.load(f"sel_dump/tmmd_{sp}.npz", allow_pickle=True)
        P = np.load(f"px_tsfm_out/tmfxt{sp}_{t}.npz")["paths"]
        for d in DS:
            e = dict(np.load(f"sel_dump/tmmdC_{d}_{sp}.npz", allow_pickle=True)); m = z["code"] == d
            assert np.array_equal(z["day"][m], e["day"])
            e["draws"] = to_draws(P[m], e["sig"].astype(np.float64), 12)
            np.savez(f"sel_dump/tmmdC_{d}_{sp}_{t}.npz", **e)
        print(t, sp, flush=True)


def fintexts(t):
    """FinTexTS: swap the candidates for TSFM samples on the rows each px_rows/fintexts_<split> file exported."""
    for sp, px in (("tr", f"ftxfxttr_{t}"), ("va", f"ftxfxtva_{t}"), ("te", f"ftxfxt_{t}")):
        e = dict(np.load(f"sel_dump/fintexts_{sp}.npz", allow_pickle=True))
        rows = np.load(f"px_rows/fintexts_{sp}.npz")["rows"]
        e = {k: v[rows] for k, v in e.items()}
        e["draws"] = to_draws(np.load(f"px_tsfm_out/{px}.npz")["paths"], e["sig"].astype(np.float64), 3)
        np.savez(f"sel_dump/fintexts_{sp}_{t}.npz", **e)
        print("fintexts", t, sp, e["draws"].shape, flush=True)


def fnspid_trval(t):
    """FNSPID stage 2: swap the train / val candidates for TSFM samples on the rows px_rows/fnspid_{tr,va}_g17 exported
    (scripts/data/export_rows_fnspid_trva.py: 30k fixed train rows, the main selector's 3,000 val rows), indices into the sorted g17 dumps."""
    for sp in ("tr", "va"):
        fs = sorted(glob.glob(f"sel_dump/fsel_fnspid_{sp}_g17_p*.npz"))
        px = np.load(f"px_rows/fnspid_{sp}_g17.npz")
        rows = px["rows"]
        assert np.all(np.diff(rows) > 0)
        parts, off = [], 0
        for f in fs:
            d = np.load(f, allow_pickle=True)
            n = len(d["day"])
            r = rows[(rows >= off) & (rows < off + n)] - off
            parts.append({k: d[k][r] for k in d.files})
            off += n
        base = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        assert len(base["day"]) == len(rows), (len(base["day"]), len(rows))
        # Same rows as the px export: the dump's 10-day truth must match the export's (up to the dump's clipped jumps).
        y = base["y"][:, :, 2].astype(np.float64) * base["sig"][:, None].astype(np.float64) * np.sqrt(np.arange(1, 11))
        ok = (np.abs(y - px["fut"]).max(1) < 0.02).mean()
        assert ok > 0.99, ok
        P = np.load(f"px_tsfm_out/fxt{sp}_{t}.npz")["paths"]
        assert P.shape[0] == len(rows) and P.shape[1] >= K and P.shape[2] == 10, P.shape
        out = dict(base, draws=to_draws(P, base["sig"].astype(np.float64), 10))
        np.savez(f"sel_dump/fsel_fnspid_{sp}_g17_{t}.npz", **out)
        print("fnspid", t, sp, out["draws"].shape, "truth match", ok, flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "fintexts":
        for t in sys.argv[2:]:
            fintexts(t)
    elif len(sys.argv) > 2 and sys.argv[1] == "fnspid":
        for t in sys.argv[2:]:
            fnspid_trval(t)
    elif len(sys.argv) > 1:
        for t in sys.argv[1:]:
            trainval(t)
    else:
        main()
