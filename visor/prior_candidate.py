"""Prior path as one candidate (Time-MMD). TaTS and MM-TSFlib read the prior_history_avg column that Time-MMD ships
(an average of earlier values of the same period) for every horizon step. Here the prior path becomes one candidate,
in the same path units as the draws (0.05 (z_{t+h} - z_t) / (sig sqrt h)):
  prior     : the last of the K=32 draws is replaced by the prior path
  priorpers : the last two draws are replaced by the prior path and the no-change path
The predictor never sees it, and the selector sees it only as one more candidate. Adopted per dataset on validation
(paper: Climate). The prior column is read from the TaTS copy of Time-MMD (set TATS_DATA).
`python -m visor.prior_candidate [src_prefix]`   (default tmmdC; reads sel_dump/<src>_<dataset>_*, writes
sel_dump/<src>prior_<dataset>_* and <src>priorpers_<dataset>_*)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from visor import paths

RAW = paths.TMMD_RAW
TATS = paths.TATS_DATA
NAME = {"Algriculture": "Agriculture", "Public_Health": "Health"}
TM = ["Algriculture", "Climate", "Economy", "Energy", "Public_Health", "Security", "SocialGood", "Traffic"]
H = 12


def main(SRC):
    for d in [x for x in TM if os.path.exists(f"sel_dump/{SRC}_{x}_va.npz")]:
        g = pd.read_csv(glob.glob(f"{RAW}/{d}/*.csv")[0])
        tt = pd.read_csv(f"{TATS}/{NAME.get(d, d)}.csv")
        assert len(g) == len(tt) and (g.date.values == tt.date.values).all(), d
        g["p"] = tt.prior_history_avg.values
        g["date"] = pd.to_datetime(g.date)
        g = g.sort_values("date").reset_index(drop=True)   # as flow_mlp.py and baselines/tats_ts.py
        ot = g.OT.ffill().bfill().to_numpy(np.float64)
        pr = pd.Series(g.p.to_numpy(np.float64)).ffill().bfill().to_numpy()
        ntr = int(len(g) * 0.7)
        mu, sd = ot[:ntr].mean(), ot[:ntr].std()
        Z, PZ = (ot - mu) / sd, (pr - mu) / sd
        idx = {}
        for i, s in enumerate(g.date):          # Health repeats a few dates: keep every index, pick by the truth below
            idx.setdefault(s.strftime("%Y-%m-%d"), []).append(i)
        sq = np.sqrt(np.arange(1, H + 1))
        for sp in ("tr", "va", "te"):
            src = f"sel_dump/{SRC}_{d}_{sp}.npz"
            z = np.load(src, allow_pickle=True)
            out = {k: z[k] for k in z.files}
            sig = z["sig"].astype(np.float64)[:, None]
            yc = z["y"][..., 2].astype(np.float64)

            def fit(r, i):
                if i + H >= len(Z):
                    return np.inf
                return np.abs(0.05 * (Z[i + 1:i + H + 1] - Z[i]) / (sig[r] * sq) - yc[r]).max()
            I = np.array([min(idx[x], key=lambda i: fit(r, i)) for r, x in enumerate(z["day"])])
            truth = 0.05 * (Z[I[:, None] + np.arange(1, H + 1)] - Z[I][:, None]) / (sig * sq)
            err = np.abs(truth - z["y"][..., 2].astype(np.float64)).max()
            assert err < 2e-3 * max(1.0, np.abs(truth).max()), (d, sp, err)   # same units and alignment as the dump
            prior = 0.05 * (PZ[I[:, None] + np.arange(1, H + 1)] - Z[I][:, None]) / (sig * sq)
            for var, cols in (("prior", [(-1, prior)]), ("priorpers", [(-1, prior), (-2, 0 * prior)])):
                dst = f"sel_dump/{SRC}{var}_{d}_{sp}.npz"
                dr = out["draws"].astype(np.float32).copy()
                for k, p in cols:
                    dr[:, k] = p[..., None].astype(np.float32)
                o2 = dict(out); o2["draws"] = dr.astype(out["draws"].dtype)
                np.savez(dst, **o2)
            print(d, sp, "rows", len(I), "truth check", f"{err:.1e}", "prior MAE vs truth (path units)",
                  f"{np.abs(prior - truth).mean():.4f}", "medoid-free mean draw MAE", f"{np.abs(out['draws'][..., 2].astype(float) - truth[:, None]).mean():.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tmmdC")
