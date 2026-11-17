"""FinTexTS windows in the selector's dump format, and the TS-to-Chart flow-matching predictor that samples their candidates.

Follows the paper's protocol where it applies to a path forecaster: 100 large US stocks, daily, 2019-2023; input 64
days, horizon 3; train / val / test = 2019-2021 / 2022 / 2023 (targets inside each split, inputs may reach back
64 days); z-score per ticker fit on the train part; errors on the normalised values. We forecast the close.

The text of a window is the dataset's own LLM summaries for the last input day, ordered target company, related
companies, sector, macro (the selector cuts it at its character limit). The generator never sees text: a conditional
flow-matching MLP (flow_mlp.Flow) is trained from scratch on the train windows of all tickers and draws K=32 paths.

The series z is stored as a log level l = S * z as in flow_mlp.py, so the selector (--data tmmd) runs on it unchanged.
Writes sel_dump/fintexts_{tr,va,te}.npz and sel_dump/fintexts_side.npz. Set VISOR_FINTEXTS_RAW to the folder of the parquet files.
`python -m visor.fintexts` (paper setting: the defaults)
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd
import torch

from visor import flow_mlp as B
from visor import paths

RAW = paths.FINTEXTS_RAW
L, H, K, S = 64, 3, 32, 0.05
TXT = [("Company", ["targetCompany_category1", "targetCompany_category2", "targetCompany_category3"]),
       ("Related companies", ["relatedCompany_category1", "relatedCompany_category2", "relatedCompany_category3"]),
       ("Sector", ["sector_category1", "sector_category2", "sector_category3"]),
       ("Macro", ["macro_category1", "macro_category2", "macro_category3"])]


def text_of(r):
    parts = []
    for name, cols in TXT:
        s = " ".join(str(r[c]).strip() for c in cols if isinstance(r[c], str) and r[c].strip() and r[c] != "nan")
        if s:
            parts.append(f"{name}: {s}")
    return "\n".join(parts)


def windows():
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{RAW}/*.parquet"))])
    df["date"] = pd.to_datetime(df.date)
    out = {s: [] for s in ("tr", "va", "te")}
    for tk, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        c = g.close.ffill().bfill().to_numpy(np.float64)
        ntr, nva = int((g.date <= "2021-12-31").sum()), int(((g.date > "2021-12-31") & (g.date <= "2022-12-31")).sum())
        n = len(g)
        z = (c - c[:ntr].mean()) / c[:ntr].std()
        b1, b2 = [0, ntr - L, ntr + nva - L], [ntr, ntr + nva, n]
        for s, lo, hi in zip(("tr", "va", "te"), b1, b2):
            for t0 in range(lo, hi - L - H + 1):
                j = t0 + L - 1
                out[s].append((tk, g.date.iloc[j].strftime("%Y-%m-%d"), z[t0:t0 + L], z[t0 + L:t0 + L + H], j, g))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--drop", type=float, default=0.1)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="fintexts")
    p.add_argument("--rep", default="nosqrt", choices=B.REPS, help="Space the generator models (flow_mlp.enc); dumps are always in the standard unit")
    a = p.parse_args()
    import os
    os.makedirs("sel_dump", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    torch.manual_seed(a.seed)
    B.L, B.H = L, H                     # flow_mlp.Flow reads the window sizes from its module globals
    W = windows()
    arr = {s: (np.array([w[2] for w in W[s]]), np.array([w[3] for w in W[s]])) for s in W}
    for s in W:
        print(s, len(W[s]), "windows", flush=True)

    hx, fx, _ = B.enc(*arr["tr"], rep=a.rep)
    fx = np.clip(fx, -30, 30)
    X, C = torch.tensor(fx, dtype=torch.float32), torch.tensor(hx, dtype=torch.float32)
    net = B.Flow(1, a.width, a.drop, a.depth)
    sd = X.std(0, keepdim=True).clamp(min=1.0)
    d0 = torch.zeros(a.bs, dtype=torch.long)

    def sample(h, k, seed, bs=4096):
        net.eval()
        out = []
        g = torch.Generator().manual_seed(seed)
        for i in range(0, len(h), bs):
            c = torch.tensor(h[i:i + bs], dtype=torch.float32).repeat_interleave(k, 0)
            x = torch.randn(len(c), H, generator=g) * sd
            dd = torch.zeros(len(c), dtype=torch.long)
            with torch.no_grad():
                for j in range(a.nfe):
                    x = x + net(x, torch.full((len(c),), j / a.nfe), c, dd) / a.nfe
            out.append(x.numpy().reshape(-1, k, H))
        net.train()
        return np.concatenate(out)

    vi = np.random.default_rng(0).choice(len(arr["va"][0]), min(5000, len(arr["va"][0])), replace=False)
    _, vf, vsig = B.norm(arr["va"][0][vi], arr["va"][1][vi])
    vh, _, vback = B.enc(arr["va"][0][vi], None, a.rep)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    best, best_state = 1e9, None
    for it in range(a.steps):
        i = torch.randint(0, len(X), (a.bs,))
        x1, c = X[i], C[i]
        x0 = torch.randn_like(x1) * sd
        t = torch.rand(len(i))
        loss = ((net((1 - t[:, None]) * x0 + t[:, None] * x1, t, c, d0) - (x1 - x0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 1000 == 999 or it == a.steps - 1:
            # Early stopping on val rows only: z-MAE of the medoid of 16 draws.
            dv = vback(sample(vh, 16, 123)) * vsig[:, None, None]
            zy = vf * vsig[:, None]
            med = np.abs(dv[:, :, None] - dv[:, None]).mean(-1).sum(-1).argmin(1)
            m = float(np.abs(dv[np.arange(len(dv)), med] - zy).mean())
            if m < best:
                best, best_state = m, {k: v.clone() for k, v in net.state_dict().items()}
            print("step", it + 1, "loss", round(loss.item(), 4), "val medoid z-MAE", round(m, 4), flush=True)
    net.load_state_dict(best_state)

    side, rep = {}, {}
    root = np.sqrt(np.arange(1, H + 1))
    for s in ("tr", "va", "te"):
        hist, fut = arr[s]
        _, f, sig = B.norm(hist, fut)
        h, _, back = B.enc(hist, None, a.rep)
        draws = back(sample(h, K, a.seed + 1))
        lsig = S * sig
        dr = (draws * S * sig[:, None, None]) / (lsig[:, None, None] * root)
        y = (f * S * sig[:, None]) / (lsig[:, None] * root)
        rows = W[s]
        np.savez(f"sel_dump/{a.out}_{s}.npz", day=np.array([r[1] for r in rows]), code=np.array([r[0] for r in rows]),
                 sig=lsig.astype(np.float32), y=np.repeat(y[..., None], 3, -1).astype(np.float16),
                 draws=np.repeat(dr[..., None], 3, -1).astype(np.float16))
        for r, hh in zip(rows, hist):
            side[(r[0], r[1])] = (hh, text_of(r[5].iloc[r[4]]))
        zc, zy = draws * sig[:, None, None], f * sig[:, None]
        med = np.abs(zc[:, :, None] - zc[:, None]).mean(-1).sum(-1).argmin(1)
        n = np.arange(len(zy))
        rep[s] = {"n": len(zy), "last": float(np.abs(zy).mean()), "medoid": float(np.abs(zc[n, med] - zy).mean()),
                  "median": float(np.abs(np.median(zc, 1) - zy).mean()), "oracle": float(np.abs(zc - zy[:, None]).mean(-1).min(1).mean()),
                  "mse_last": float((zy ** 2).mean()), "mse_median": float(((np.median(zc, 1) - zy) ** 2).mean())}
        print(s, rep[s], flush=True)
    keys = list(side)
    if a.rep == "nosqrt":  # the side file does not depend on the generator; rep variants reuse fintexts_side.npz
        np.savez(f"sel_dump/{a.out}_side.npz", code=np.array([k[0] for k in keys]), day=np.array([k[1] for k in keys]),
             hist=np.array([side[k][0] for k in keys]), text=np.array([side[k][1] for k in keys], dtype=object))
    torch.save(net.state_dict(), f"sel_dump/{a.out}_gen.pt")
    json.dump({"args": vars(a), "best_val": best, "zmae": rep}, open(f"results/{a.out}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
