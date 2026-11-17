"""Time-MMD TS-to-Chart predictor with the benchmark prior, a ridge backbone and out-of-fold training candidates.

Same windows, splits, units and dump format as flow_mlp.py (imported, unchanged), one model per dataset:
  --level 1  condition on the dataset-standardized level of the history (last value, mean, std in z units, log sigma);
             the instance scaling removes the level, which mean-reverting or seasonal series need.
  --lin 1    a ridge forecaster in the normalized space; the flow samples the residual around it (point backbone +
             stochastic residual). Ridge lambda via --ridge; --resid_scale multiplies the residual noise prior.
  --prior 1  the predictor is also conditioned on the prior_history_avg column that Time-MMD ships for every step and
             that the multimodal baselines (TaTS, MM-TSFlib) read for the horizon. It is a number built from earlier
             values of the same series, not text, so the predictor stays text-free. Given in norm units,
             (prior_{t+h} - x_t) / sig.
  --oof k    the train rows get out-of-fold draws: the train span is cut into k contiguous blocks, and each block is
             sampled by a model fit on the other blocks (rows whose window overlaps the block are dropped from that
             fit). Without it, train candidates come from a model that saw those very rows and are far closer to the
             truth than on validation or test. Validation and test draws come from the model fit on all train rows.
  --refit_val 1  test draws come from a model refit on train+val rows for as many steps as the full-train fit's best
             val step (last state kept); also a val-only recency check. --refit_dom limits it to some datasets.
Model selection uses validation only. Writes sel_dump/<out>_<dataset>_{tr,va,te}.npz and results/<out>.json.
`python -m visor.flow_mlp_prior --out tmmdP_A --prior 1 --lin 1 --oof 5 [--dom Energy]`
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from visor import flow_mlp as B

L, H, S = B.L, B.H, B.S


class Flow(nn.Module):
    """Velocity MLP of the flow over the H future steps, conditioned on the history features c."""

    def __init__(self, cin, w, drop, depth):
        super().__init__()
        layers, d = [], H + cin + 1
        for j in range(depth):
            layers += [nn.Linear(d, w), nn.SiLU()] + ([nn.Dropout(drop)] if drop > 0 else [])
            d = w
        self.net = nn.Sequential(*layers, nn.Linear(w, H))

    def forward(self, x, t, c):
        return self.net(torch.cat([x, c, t[:, None]], 1))


def windows_prior():
    """Same rows and order as flow_mlp.windows() (asserted below), plus the prior of the horizon steps in z units."""
    out = {s: [] for s in ("tr", "va", "te")}
    for d in B.DOM:
        g = pd.read_csv(glob.glob(f"{B.RAW}/{d}/*.csv")[0])
        g["date"] = pd.to_datetime(g.date)
        g = g.sort_values("date").reset_index(drop=True)
        ot = g.OT.ffill().bfill().to_numpy(np.float64)
        pr = g.prior_history_avg.ffill().bfill().to_numpy(np.float64)
        n = len(g)
        ntr, nte = int(n * 0.7), int(n * 0.2)
        nva = n - ntr - nte
        mu, sd = ot[:ntr].mean(), ot[:ntr].std()
        z, pz = (ot - mu) / sd, (pr - mu) / sd
        for s, lo, hi in zip(("tr", "va", "te"), [0, ntr - L, n - nte - L], [ntr, ntr + nva, n]):
            for t0 in range(lo, hi - L - H + 1):
                out[s].append((d, g.date.iloc[t0 + L - 1].strftime("%Y-%m-%d"), z[t0:t0 + L], z[t0 + L:t0 + L + H], pz[t0 + L:t0 + L + H], t0))
    return out


def feats(hist, prior, level, use_prior):
    h, _, sig = B.norm(hist)
    c = [h]
    if level:
        c += [hist[:, -1:], hist.mean(1, keepdims=True), hist.std(1, keepdims=True), np.log(sig)[:, None]]
    if use_prior:
        c += [(prior - hist[:, -1:]) / sig[:, None]]
    return np.concatenate(c, 1), sig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--dom", default="", help="comma list; default all eight")
    p.add_argument("--prior", type=int, default=1)
    p.add_argument("--oof", type=int, default=0)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--lin", type=int, default=1)
    p.add_argument("--ridge", type=float, default=10.0)
    p.add_argument("--resid_scale", type=float, default=1.0)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--drop", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--clip", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--refit_val", type=int, default=0, help="test draws from the predictor refit on train+val (see docstring)")
    p.add_argument("--refit_dom", default="", help="comma list of datasets to refit; default every processed dataset")
    a = p.parse_args()
    os.makedirs("sel_dump", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(4)
    W0, W = B.windows(), windows_prior()
    for s in W:   # identical rows and order as flow_mlp.windows()
        assert len(W[s]) == len(W0[s]) and all(u[0] == v[0] and u[1] == v[1] and np.allclose(u[2], v[2]) for u, v in zip(W[s], W0[s])), s
    doms = a.dom.split(",") if a.dom else list(B.DOM)
    root = np.sqrt(np.arange(1, H + 1))
    rep = {}
    rdoms = (a.refit_dom.split(",") if a.refit_dom else doms) if a.refit_val else []
    assert all(d in doms for d in rdoms), rdoms
    for d in doms:
        rows = {s: [w for w in W[s] if w[0] == d] for s in ("tr", "va", "te")}
        hist = {s: np.array([w[2] for w in rows[s]]) for s in rows}
        fut = {s: np.array([w[3] for w in rows[s]]) for s in rows}
        pri = {s: np.array([w[4] for w in rows[s]]) for s in rows}
        t0 = {s: np.array([w[5] for w in rows[s]]) for s in rows}
        _, vf, _ = B.norm(hist["va"], fut["va"])
        vy = vf / root

        def fit(m, extra=None, fixed_steps=None):
            """Flow (+ ridge) on the train rows in mask m, early stopping on validation. Returns sample(hist, prior, k, seed).
            extra=(hist, fut, prior) adds rows (the refit on val); fixed_steps trains that long and keeps the last state."""
            hh, ff, pp = hist["tr"][m], fut["tr"][m], pri["tr"][m]
            if extra is not None:
                hh, ff, pp = np.r_[hh, extra[0]], np.r_[ff, extra[1]], np.r_[pp, extra[2]]
            C, _ = feats(hh, pp, a.level, a.prior)
            _, F, _ = B.norm(hh, ff)
            F = np.clip(F, -a.clip, a.clip)
            mu_c, sd_c = C.mean(0), C.std(0) + 1e-6
            Cn = (C - mu_c) / sd_c
            if a.lin:
                X1 = np.c_[Cn, np.ones(len(Cn))]
                Wr = np.linalg.solve(X1.T @ X1 + a.ridge * np.eye(X1.shape[1]), X1.T @ F)
                lin = lambda cn: np.c_[cn, np.ones(len(cn))] @ Wr
            else:
                lin = lambda cn: np.zeros((len(cn), H))
            X = torch.tensor(F - lin(Cn), dtype=torch.float32)
            Ct = torch.tensor(Cn, dtype=torch.float32)
            sd = X.std(0, keepdim=True).clamp(min=0.05 if a.lin else 1.0) * a.resid_scale
            net = Flow(Cn.shape[1], a.width, a.drop, a.depth)

            def sample(h, pr, k, seed):
                c, _ = feats(h, pr, a.level, a.prior)
                cn = (c - mu_c) / sd_c
                net.eval()
                ct = torch.tensor(cn, dtype=torch.float32).repeat_interleave(k, 0)
                g = torch.Generator().manual_seed(seed)
                x = torch.randn(len(ct), H, generator=g) * sd
                with torch.no_grad():
                    for j in range(a.nfe):
                        x = x + net(x, torch.full((len(ct),), j / a.nfe), ct) / a.nfe
                net.train()
                return x.numpy().reshape(len(h), k, H) + lin(cn)[:, None]

            def val_medoid():
                dv = sample(hist["va"], pri["va"], 16, 123) / root
                n = np.arange(len(dv))
                med = np.abs(dv[:, :, None] - dv[:, None]).mean(-1).sum(-1).argmin(1)
                return float(np.abs(dv[n, med] - vy).mean())

            nst = fixed_steps or a.steps
            opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=a.wd)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nst)
            best, state, best_it = 1e9, None, 0
            every = max(a.steps // 20, 200)
            for it in range(nst):
                i = torch.randint(0, len(X), (a.bs,))
                x1, c = X[i], Ct[i]
                x0 = torch.randn_like(x1) * sd
                t = torch.rand(len(i))
                loss = ((net((1 - t[:, None]) * x0 + t[:, None] * x1, t, c) - (x1 - x0)) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sch.step()
                if fixed_steps is None and ((it + 1) % every == 0 or it == a.steps - 1):
                    v = val_medoid()
                    if v < best:
                        best, state, best_it = v, {k: v_.clone() for k, v_ in net.state_dict().items()}, it + 1
            if fixed_steps is not None:
                return sample, val_medoid()      # last state; val is in its training rows here (logged only)
            net.load_state_dict(state)
            FITSTEP[0] = best_it
            return sample, best

        FITSTEP = [0]
        full, vbest = fit(np.ones(len(rows["tr"]), bool))
        full_step = FITSTEP[0]
        dr = {s: full(hist[s], pri[s], a.K, a.seed + 1) for s in ("va", "te")}
        if a.oof > 1:
            # Contiguous time blocks of the train rows; a fit for block b drops every row whose window [t0, t0+L+H)
            # overlaps the block's span, so no target of the block is ever seen in training.
            n = len(rows["tr"])
            edges = np.linspace(0, n, a.oof + 1).astype(int)
            out = np.zeros((n, a.K, H), np.float64)
            for b in range(a.oof):
                blk = np.arange(edges[b], edges[b + 1])
                lo, hi = t0["tr"][blk].min(), t0["tr"][blk].max() + L + H
                keep = (t0["tr"] + L + H <= lo) | (t0["tr"] >= hi)
                smp, _ = fit(keep)
                out[blk] = smp(hist["tr"][blk], pri["tr"][blk], a.K, a.seed + 1 + b)
            dr["tr"] = out
        else:
            dr["tr"] = full(hist["tr"], pri["tr"], a.K, a.seed + 1)
        rep[d] = {"val_select_medoid": vbest}
        if d in rdoms:
            rng = torch.get_rng_state()
            ntr_, nv = len(rows["tr"]), len(rows["va"])
            # 1) val-only recency check: without the last nv train rows vs the full-train fit, both early-stopped on val
            _, v_old = fit(np.arange(ntr_) < max(ntr_ - nv, 1))
            # 2) the refit: train + val rows, as many steps as the full-train fit's best val step, last state kept
            smp, v_after = fit(np.ones(ntr_, bool), extra=(hist["va"], fut["va"], pri["va"]), fixed_steps=max(full_step, 1))
            dr["te"] = smp(hist["te"], pri["te"], a.K, a.seed + 1)
            torch.set_rng_state(rng)
            rep[d]["recency"] = dict(val_full=vbest, val_without_recent=v_old, refit_step=full_step, val_after_refit=v_after)
            print(a.out, d, "recency", rep[d]["recency"], flush=True)
        for s in ("tr", "va", "te"):
            _, f, sig = B.norm(hist[s], fut[s])
            drv, y = dr[s] / root, f / root
            np.savez(f"sel_dump/{a.out}_{d}_{s}.npz", day=np.array([r[1] for r in rows[s]]), code=np.array([r[0] for r in rows[s]]),
                     sig=(S * sig).astype(np.float32), y=np.repeat(y[..., None], 3, -1).astype(np.float16),
                     draws=np.repeat(drv[..., None], 3, -1).astype(np.float16))
            n = np.arange(len(y))
            med = np.abs(drv[:, :, None] - drv[:, None]).mean(-1).sum(-1).argmin(1)
            rep[d][s] = dict(n=len(y), medoid=float(np.abs(drv[n, med] - y).mean()), median=float(np.abs(np.median(drv, 1) - y).mean()),
                             oracle=float(np.abs(drv - y[:, None]).mean(-1).min(1).mean()), mean_draw=float(np.abs(drv - y[:, None]).mean()))
        r = rep[d]
        print(f"{a.out} {d:14s} medoid tr {r['tr']['medoid']:.3f} va {r['va']['medoid']:.3f} | mean draw tr {r['tr']['mean_draw']:.3f} "
              f"va {r['va']['mean_draw']:.3f} | oracle va {r['va']['oracle']:.3f}", flush=True)
        if a.refit_dom and rdoms and d == max(rdoms, key=doms.index):
            break
    json.dump({"args": vars(a), "stats": rep}, open(f"results/{a.out}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
