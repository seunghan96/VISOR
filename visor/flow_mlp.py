"""Time-MMD (Liu et al., NeurIPS 2024 D&B) windows in the selector's dump format, with a text-free stochastic generator.

Follows MM-TSFlib's preprocessed files and protocol where it can: target OT, each domain split 70/10/20 in time, a
StandardScaler fit on the train part, input length 24, text = Final_Search_4 of the last input step. The horizon is 12,
which is one of the official horizons for both monthly (6, 8, 10, 12) and weekly (12, 24, 36, 48) domains. Rows are
sorted by date (Economy ships in reverse order). Environment (daily, .rar only) is left out.

The series z is stored as a log level l = S * z (S small) so the selector code, which works on log prices, runs as is;
metrics are reported back in z units. A conditional flow-matching MLP (the TS-to-Chart predictor of VISOR), trained
from scratch on numbers only and never on text, draws K=32 future paths per window. With --per_dom 1 (paper setting)
each dataset gets its own flow; --pre auto first pretrains each flow on price windows dated before the dataset's
validation period (scripts/data/make_stock_windows.py); --oof 5 draws the training candidates out of fold.

Writes sel_dump/<out>_{tr,va,te}.npz (day, code, sig, y, draws in the FNSPID dump convention; all eight datasets in one
file, see visor/pools.py slice), sel_dump/<out>_gen.pt (the flows) and sel_dump/tmmd_side.npz (per-row history and
text, read by the selector; written by the first run or by --out tmmd, identical for every configuration).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from visor import paths

RAW = paths.TMMD_RAW
L, H, K, S = 24, 12, 32, 0.05
DOM = {"Algriculture": "Agriculture: US retail broiler composite price (monthly)", "Climate": "Climate: US precipitation (monthly)",
       "Economy": "Economy: US trade balance (monthly)", "Energy": "Energy: US gasoline price (weekly)",
       "Public_Health": "Public health: US influenza-like illness ratio (weekly)", "Security": "Security: US FEMA grants (monthly)",
       "SocialGood": "Social good: US unemployment rate, unadjusted (monthly)", "Traffic": "Traffic: US vehicle miles traveled (monthly)"}


def windows():
    out = {s: [] for s in ("tr", "va", "te")}
    out["extra"] = []
    for d in DOM:
        f = glob.glob(f"{RAW}/{d}/*.csv")[0]
        g = pd.read_csv(f)
        g["date"] = pd.to_datetime(g.date)
        g = g.sort_values("date").reset_index(drop=True)
        ot = g.OT.ffill().bfill().to_numpy(np.float64)
        txt = g["Final_Search_4"].fillna("").astype(str).to_numpy()
        n = len(g)
        ntr, nte = int(n * 0.7), int(n * 0.2)
        nva = n - ntr - nte
        z = (ot - ot[:ntr].mean()) / ot[:ntr].std()
        # Other numeric columns of the same file, train part only, as extra generator training data (numbers, no text).
        for col in g.columns:
            if col in ("OT", "date", "start_date", "end_date", "prior_history_avg", "prior_history_std") or not pd.api.types.is_numeric_dtype(g[col]):
                continue
            v = g[col].to_numpy(np.float64)[:ntr]
            if np.isnan(v).mean() > 0.2 or np.nanstd(v) == 0:
                continue
            v = pd.Series(v).ffill().bfill().to_numpy()
            v = (v - v.mean()) / v.std()
            for t0 in range(0, ntr - L - H + 1):
                out["extra"].append((d, "", v[t0:t0 + L], v[t0 + L:t0 + L + H], ""))
        # MM-TSFlib borders: each split's inputs may start seq_len before it; targets stay inside it.
        b1, b2 = [0, ntr - L, n - nte - L], [ntr, ntr + nva, n]
        for s, lo, hi in zip(("tr", "va", "te"), b1, b2):
            for t0 in range(lo, hi - L - H + 1):
                out[s].append((d, g.date.iloc[t0 + L - 1].strftime("%Y-%m-%d"), z[t0:t0 + L], z[t0 + L:t0 + L + H], txt[t0 + L - 1]))
    return out


def norm(hist, fut=None):
    """Instance scale: last value as origin, std of the input's steps as unit (as sig in the FNSPID dumps)."""
    sig = np.maximum(np.diff(hist, axis=-1).std(-1), 1e-3)
    h = (hist - hist[..., -1:]) / sig[..., None]
    f = None if fut is None else (fut - hist[..., -1:]) / sig[..., None]
    return h, f, sig


REPS = ("nosqrt", "sqrt", "zscore", "robust")


def enc(hist, fut=None, rep="nosqrt"):
    """Path representation the generator models (ablation). Returns (cond, target, back): back maps samples
    in the modeled space to norm()'s unit (x_{t+h} - x_t) / sig, so dumps stay in one unit whatever the rep.
      nosqrt  (x_{t+h} - x_t) / sig, cond = norm's history: exactly what the generator always modeled (default)
      sqrt    (x_{t+h} - x_t) / (sig sqrt(h)), same cond
      zscore  (x_{t+h} - mean(hist)) / std(hist), cond = z-scored history (no last-value differencing)"""
    if rep == "nosqrt":
        h, f, _ = norm(hist, fut)
        return h, f, lambda x: x
    if rep == "sqrt":
        h, f, _ = norm(hist, fut)
        root = np.sqrt(np.arange(1, H + 1))  # module H (fintexts.py sets B.H before use)
        return h, None if f is None else f / root, lambda x: x * root
    if rep == "zscore":
        sig = np.maximum(np.diff(hist, axis=-1).std(-1), 1e-3)
        m, s = hist.mean(-1, keepdims=True), np.maximum(hist.std(-1, keepdims=True), 1e-3)
        c = (hist - m) / s
        f = None if fut is None else (fut - m) / s
        last = hist[..., -1:]
        # x: (n, k, H) or (n, H) samples in z-score units -> (x_{t+h} - x_t) / sig
        back = lambda x: ((m + s * x) - last) / sig[:, None] if x.ndim == 2 else ((m[:, None] + s[:, None] * x) - last[:, None]) / sig[:, None, None]
        return c, f, back
    if rep == "robust":
        # (a past spike inflates the std of the input's steps) the flow sees history and future
        # scaled by a robust step scale (1.4826 x MAD of the steps) instead of their std; dumps keep norm()'s unit, so
        # the evaluation units (and every baseline's) are unchanged.
        dh = np.diff(hist, axis=-1)
        sig = np.maximum(dh.std(-1), 1e-3)
        rs = np.maximum(1.4826 * np.median(np.abs(dh - np.median(dh, -1, keepdims=True)), -1), 0.05 * sig)
        last = hist[..., -1:]
        c = (hist - last) / rs[..., None]
        f = None if fut is None else (fut - last) / rs[..., None]
        ratio = rs / sig
        back = lambda x: x * (ratio[:, None] if x.ndim == 2 else ratio[:, None, None])
        return c, f, back
    raise ValueError(rep)


class Flow(nn.Module):
    def __init__(self, nd, w=256, drop=0.1, depth=3):
        super().__init__()
        self.emb = nn.Embedding(nd, 16)
        mid = []
        for j in range(depth - 1):
            mid += [nn.Linear(w, w), nn.SiLU()] + ([nn.Dropout(drop)] if j < depth - 2 else [])
        self.net = nn.Sequential(nn.Linear(H + L + 16 + 1, w), nn.SiLU(), nn.Dropout(drop), *mid, nn.Linear(w, H))

    def forward(self, x, t, c, d):
        return self.net(torch.cat([x, c, self.emb(d), t[:, None]], 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--drop", type=float, default=0.1)
    p.add_argument("--extra", type=int, default=1, help="Also train on the files' other numeric columns (train part)")
    p.add_argument("--nodom", type=int, default=1, help="Hide the domain id from the generator")
    p.add_argument("--depth", type=int, default=3, help="Hidden layers of the flow MLP")
    p.add_argument("--nfe", type=int, default=32, help="Euler steps when sampling")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--pre", default="", help="Pretrain on stock windows first: 'Dataset=tag,...' (sel_dump/stockwin_<tag>.npz, scripts/data/make_stock_windows.py); 'auto' maps weekly datasets to W2015, Social Good to M2002, the rest to M2010")
    p.add_argument("--pre_steps", type=int, default=30000)
    p.add_argument("--per_dom", type=int, default=0, help="Train one flow per domain instead of one shared flow")
    p.add_argument("--out", default="tmmd", help="Dump prefix")
    p.add_argument("--rep", default="nosqrt", choices=REPS, help="Space the generator models (see enc); dumps are always in the standard unit")
    p.add_argument("--refit_val", type=int, default=0, help="Test draws from a flow per domain refit on train+val windows for as many steps as the first fit's best val step. Also runs the val-only recency check: a flow fit without the last stretch of train (as long as val) vs the full-train flow, both scored on val. Train/val draws unchanged.")
    p.add_argument("--oof", type=int, default=0, help="Out-of-fold train draws (needs --per_dom 1): k time blocks per domain, each sampled by a flow fit without the rows whose window overlaps the block. Val/test draws unchanged.")
    a = p.parse_args()
    os.makedirs("sel_dump", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    torch.manual_seed(a.seed)
    W = windows()
    dom = {d: i for i, d in enumerate(DOM)}
    arr = {s: (np.array([w[2] for w in W[s]]), np.array([w[3] for w in W[s]]), np.array([dom[w[0]] for w in W[s]])) for s in W}
    for s in W:
        print(s, len(W[s]), "windows", flush=True)
    trh, trf, trd = arr["tr"]
    if a.extra and W["extra"]:
        trh = np.r_[trh, np.array([w[2] for w in W["extra"]])]
        trf = np.r_[trf, np.array([w[3] for w in W["extra"]])]
        trd = np.r_[trd, np.array([dom[w[0]] for w in W["extra"]])]
        print("extra windows", len(W["extra"]), flush=True)
    rawd = trd.copy()

    if a.pre == "auto":
        a.pre = "Energy=W2015,Public_Health=W2015,SocialGood=M2002," + ",".join(f"{d}=M2010" for d in ("Algriculture", "Climate", "Economy", "Security", "Traffic"))
    pre_map = dict(kv.split("=") for kv in a.pre.split(",")) if a.pre else {}
    pre_state = {}

    def pretrain(tag):
        """One flow trained on stock windows (numbers only), cached per corpus."""
        if tag not in pre_state:
            z = np.load(f"sel_dump/stockwin_{tag}.npz")
            hx, fx, _ = enc(z["hist"].astype(np.float64), z["fut"].astype(np.float64), a.rep)
            X, C = torch.tensor(np.clip(fx, -30, 30), dtype=torch.float32), torch.tensor(hx, dtype=torch.float32)
            net = Flow(len(DOM), a.width, a.drop, a.depth)
            opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.pre_steps)
            d0 = torch.zeros(a.bs, dtype=torch.long)
            for it in range(a.pre_steps):
                i = torch.randint(0, len(X), (a.bs,))
                x1, c = X[i], C[i]
                x0 = torch.randn_like(x1) * x1.std(0, keepdim=True).clamp(min=1.0)
                t = torch.rand(len(i))
                loss = ((net((1 - t[:, None]) * x0 + t[:, None] * x1, t, c, d0) - (x1 - x0)) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                if it % 5000 == 4999:
                    print("pretrain", tag, it + 1, round(loss.item(), 4), flush=True)
            pre_state[tag] = {k: v.clone() for k, v in net.state_dict().items()}
        return pre_state[tag]

    def fit(mtr, mva, tag, extra_hf=None, fixed_steps=None):
        """Train one flow on the train rows in mtr; early stopping on the val rows in mva. Returns its sampler.
        extra_hf=(hist, fut, dom) adds windows (refit on val); fixed_steps trains that long and keeps the last state."""
        if extra_hf is None:
            hx, fx, _ = enc(trh[mtr], trf[mtr], a.rep)
        else:
            hx, fx, _ = enc(np.r_[trh[mtr], extra_hf[0]], np.r_[trf[mtr], extra_hf[1]], a.rep)
        # Clip extreme normalised moves so a few jumps do not dominate the flow's training.
        fx = np.clip(fx, -30, 30)
        rd = rawd[mtr] if extra_hf is None else np.r_[rawd[mtr], extra_hf[2]]
        dd0 = np.zeros_like(rd) if a.nodom else rd
        X, C, Dd = (torch.tensor(v, dtype=torch.float32) for v in (fx, hx, dd0))
        Dd = Dd.long()
        net = Flow(len(DOM), a.width, a.drop, a.depth)
        ptag = tag.removesuffix("_old").removesuffix("_refit")
        if ptag in pre_map:
            net.load_state_dict(pretrain(pre_map[ptag]))
        sd = X.std(0, keepdim=True).clamp(min=1.0)

        def sample(h, d, k, seed):
            net.eval()
            c = torch.tensor(h, dtype=torch.float32).repeat_interleave(k, 0)
            dd = torch.tensor(np.zeros_like(d) if a.nodom else d).long().repeat_interleave(k, 0)
            g = torch.Generator().manual_seed(seed)
            x = torch.randn(len(c), H, generator=g) * sd
            with torch.no_grad():
                for j in range(a.nfe):
                    x = x + net(x, torch.full((len(c),), j / a.nfe), c, dd) / a.nfe
            net.train()
            return x.numpy().reshape(len(h), k, H)

        _, vf, vsig = norm(arr["va"][0][mva], arr["va"][1][mva])
        vh, _, vback = enc(arr["va"][0][mva], None, a.rep)
        best, best_state = 1e9, None

        def val_mae():
            dv = vback(sample(vh, arr["va"][2][mva], 16, 123)) * vsig[:, None, None]
            zy = vf * vsig[:, None]
            med = np.abs(dv[:, :, None] - dv[:, None]).mean(-1).sum(-1).argmin(1)
            return float(np.abs(dv[np.arange(len(dv)), med] - zy).mean())

        if ptag in pre_map and fixed_steps is None:
            # The pretrained flow itself is a candidate too, so fine-tuning can only help on val.
            best, best_state = val_mae(), {k: v.clone() for k, v in net.state_dict().items()}
            print(tag, "pretrained only val medoid z-MAE", round(best, 4), flush=True)
        opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
        nst = fixed_steps or a.steps
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nst)
        best_it = 0
        for it in range(nst):
            i = torch.randint(0, len(X), (a.bs,))
            x1, c, d = X[i], C[i], Dd[i]
            x0 = torch.randn_like(x1) * x1.std(0, keepdim=True).clamp(min=1.0)
            t = torch.rand(len(i))
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1
            loss = ((net(xt, t, c, d) - (x1 - x0)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if fixed_steps is None and (it % 1000 == 999 or it == a.steps - 1):
                # Early stopping on val: z-MAE of the medoid of 16 draws (no text, no test rows involved).
                dv = vback(sample(vh, arr["va"][2][mva], 16, 123)) * vsig[:, None, None]
                zy = vf * vsig[:, None]
                med = np.abs(dv[:, :, None] - dv[:, None]).mean(-1).sum(-1).argmin(1)
                m = np.abs(dv[np.arange(len(dv)), med] - zy).mean()
                if m < best:
                    best, best_state, best_it = m, {k: v.clone() for k, v in net.state_dict().items()}, it + 1
                print(tag, "step", it + 1, "loss", round(loss.item(), 4), "val medoid z-MAE", round(float(m), 4), flush=True)
        if fixed_steps is None:
            net.load_state_dict(best_state)
            print(tag, "best val medoid z-MAE", round(float(best), 4), "at step", best_it, flush=True)
            FIT[tag] = dict(best=float(best), step=max(best_it, 1000))
        else:
            FIT[tag] = dict(val_after_refit=val_mae())
        nets.append(net)
        FIT[tag].update(net_idx=len(nets) - 1, sd=sd.numpy().tolist())
        return sample

    nets, FIT = [], {}
    if a.per_dom:
        # One flow per domain, as in the Time-MMD / MM-TSFlib protocol (each dataset trained on its own).
        samplers = {i: fit(rawd == i, arr["va"][2] == i, d) for d, i in dom.items()}
        FULL = {d: dict(FIT[d]) for d in dom}
    else:
        f_all = fit(np.ones(len(trh), bool), np.ones(len(arr["va"][0]), bool), "all")
        samplers = {i: f_all for i in dom.values()}

    def sample(h, d, k, seed):
        out = np.zeros((len(h), k, H), np.float32)
        for i in np.unique(d):
            m = d == i
            out[m] = samplers[i](h[m], d[m], k, seed)
        return out

    oof_tr = None
    if a.oof > 1:
        # Train candidates as the selector meets them on val/test: drawn by a model that never saw their targets.
        assert a.per_dom, "--oof needs --per_dom 1"
        hist_tr, _, d_tr = arr["tr"]
        h_tr, _, _ = enc(hist_tr, None, a.rep)
        oof_tr = np.zeros((len(hist_tr), K, H), np.float32)
        for dn, i in dom.items():
            pos = np.where(d_tr == i)[0]          # this domain's train rows, window start t0 = 0..n-1 in order
            n = len(pos)
            edges = np.linspace(0, n, a.oof + 1).astype(int)
            j = np.arange(n)
            for b in range(a.oof):
                blk = np.arange(edges[b], edges[b + 1])
                lo, hi = blk.min(), blk.max() + L + H
                keep = (j + L + H <= lo) | (j >= hi)
                m = rawd == i                        # keeps this domain's extra windows (other columns, train part)
                m[:len(hist_tr)] = False
                m[pos[keep]] = True
                smp = fit(m, arr["va"][2] == i, f"{dn}")
                oof_tr[pos[blk]] = smp(h_tr[pos[blk]], d_tr[pos[blk]], K, a.seed + 1 + b)
                print("oof", dn, "block", b, "rows", len(blk), "fit rows", int(m.sum()), flush=True)

    te_samplers, recency = {}, {}
    if a.refit_val:
        assert a.per_dom, "--refit_val needs --per_dom 1"
        hist_tr, _, d_tr = arr["tr"]
        for dn, i in dom.items():
            # 1) val-only recency check: drop the train windows whose targets reach into the last stretch of train
            #    (as many windows as val has), refit with the same early stopping, compare val with the full-train flow.
            pos = np.where(d_tr == i)[0]
            nv = int((arr["va"][2] == i).sum())
            m = rawd == i
            m[:len(hist_tr)] = False
            m[pos[:max(len(pos) - nv, 1)]] = True
            fit(m, arr["va"][2] == i, f"{dn}_old")
            recency[dn] = dict(val_full=FULL[dn]["best"], val_without_recent=FIT[f"{dn}_old"]["best"])
            # 2) the refit: train + val windows, as many steps as the full-train flow's best val step
            mv = arr["va"][2] == i
            te_samplers[i] = fit(rawd == i, mv, f"{dn}_refit", extra_hf=(arr["va"][0][mv], arr["va"][1][mv], arr["va"][2][mv]),
                                 fixed_steps=FULL[dn]["step"])
            print("recency", dn, recency[dn], flush=True)

    side, rep = {}, {}
    for s in ("tr", "va", "te"):
        hist, fut, d = arr[s]
        _, f, sig = norm(hist, fut)
        h, _, back = enc(hist, None, a.rep)
        if s == "te" and te_samplers:
            raw = np.zeros((len(h), K, H), np.float32)
            for i in np.unique(d):
                mm = d == i
                raw[mm] = te_samplers[i](h[mm], d[mm], K, a.seed + 1)
            draws = back(raw)
        else:
            draws = back(oof_tr) if (s == "tr" and oof_tr is not None) else back(sample(h, d, K, a.seed + 1))
        # FNSPID convention: log-level change / (sig * sqrt(h)), the same value in the high, low and close slots.
        lsig = S * sig
        root = np.sqrt(np.arange(1, H + 1))
        dr = (draws * S * sig[:, None, None]) / (lsig[:, None, None] * root)
        y = (f * S * sig[:, None]) / (lsig[:, None] * root)
        rows = W[s]
        np.savez(f"sel_dump/{a.out}_{s}.npz", day=np.array([r[1] for r in rows]), code=np.array([r[0] for r in rows]),
                 sig=lsig.astype(np.float32), y=np.repeat(y[..., None], 3, -1).astype(np.float16),
                 draws=np.repeat(dr[..., None], 3, -1).astype(np.float16))
        for r, hh in zip(rows, hist):
            side[(r[0], r[1])] = (hh, r[4])
        # Quick sanity: candidate spread and naive errors in z units.
        zc = draws * sig[:, None, None]
        zy = f * sig[:, None]
        med = np.abs(zc[:, :, None] - zc[:, None]).mean(-1).sum(-1).argmin(1)
        n = np.arange(len(zy))
        for i, dn in enumerate(DOM):
            m = d == i
            if m.any():
                rep.setdefault(s, {})[dn] = {"n": int(m.sum()), "last": float(np.abs(zy[m]).mean()), "medoid": float(np.abs(zc[n, med] - zy)[m].mean()),
                                             "median": float(np.abs(np.median(zc, 1) - zy)[m].mean()),
                                             "oracle": float(np.abs(zc - zy[:, None]).mean(-1).min(1)[m].mean())}
        rep.setdefault(s, {})["all"] = {"last": float(np.abs(zy).mean()), "medoid": float(np.abs(zc[n, med] - zy).mean()),
                                        "median": float(np.abs(np.median(zc, 1) - zy).mean()), "oracle": float(np.abs(zc - zy[:, None]).mean(-1).min(1).mean())}
        print(s, "z-MAE last-value %.4f  medoid %.4f  median %.4f  sample %.4f  oracle %.4f" % (
            np.abs(zy).mean(), np.abs(zc[n, med] - zy).mean(), np.abs(np.median(zc, 1) - zy).mean(),
            np.abs(zc[:, 0] - zy).mean(), np.abs(zc - zy[:, None]).mean(-1).min(1).mean()), flush=True)
    torch.save([n.state_dict() for n in nets], f"sel_dump/{a.out}_gen.pt")
    import json
    json.dump({"args": vars(a), "zmae": rep, "fit": FIT, "recency": recency}, open(f"results/{a.out}.json", "w"), indent=1)
    if a.out != "tmmd" and os.path.exists("sel_dump/tmmd_side.npz"):
        return
    keys = list(side)
    np.savez("sel_dump/tmmd_side.npz", code=np.array([k[0] for k in keys]), day=np.array([k[1] for k in keys]),
             hist=np.array([side[k][0] for k in keys]), text=np.array([side[k][1] for k in keys], dtype=object))


if __name__ == "__main__":
    main()
