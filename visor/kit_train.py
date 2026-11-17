"""Train the Chart-Native predictor of VISOR on FNSPID: a KiT-architecture flow model trained from scratch.

Training follows KiT's config: rectified flow, EMA weights, AdamW with weight decay, and the three condition dropouts.
Validation uses anchors after --train_end and KiT's metric definitions; the best EMA checkpoint is mf_runs/<tag>.pt.
Paper setting (configs/fnspid.json): --mode kit --chart 96 --pix 64 --pix_z 4.0 --win 90/10/40 --d_model 256
--n_layers 6 --steps 30000 --batch 128 --lr 2e-4 --train_end 2017-12-31 (second candidate set: 2021-12-31).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from visor.kit_flow import (KIT, SCALES, WINDOWS, Mem, bars_day, build_model, candle_path, collate, decode, load_kit_weights, measure, render,
                   pix_expect, pix_numeric, pix_probs, pix_target, pix_y, scale_tensor, to_minutes, vol_unit)

import sys  # noqa: E402
sys.path.insert(0, KIT)
from kit_v1.data.normalize import load_stats  # noqa: E402
from kit_v1.infer.sampler import make_nohist_cond  # noqa: E402

STATS = load_stats(f"{KIT}/ckpt/hf/stats.json")
FEATS = ("clock", "sess", "dow", "event", "month", "yday", "market", "sector", "inst", "scale", "pad_mask", "tok_scale", "unit")


def save_result(name, obj):
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump(obj, f, indent=1)
    print(name, json.dumps(obj), flush=True)


class Windows(IterableDataset):
    def __init__(self, a, seed):
        self.a, self.seed = a, seed
        self.markets = [m.split(":")[0] for m in a.market_w.split(",")]
        self.w = np.array([float(m.split(":")[1]) for m in a.market_w.split(",")])
        self.w = self.w / self.w.sum()
        self.scales = a.scales.split(",")

    def __iter__(self):
        a = self.a
        info = torch.utils.data.get_worker_info()
        rs = np.random.default_rng(self.seed + 1000 * (info.id if info else 0))
        mem = Mem(self.markets, self.scales)
        cut = to_minutes(a.train_end) + 1440
        tab = {}
        for key, ser in mem.s.items():
            hi = np.array([ser.upto(i, cut) for i in range(len(ser.codes))])
            n = np.maximum(hi - ser.start - WINDOWS[key[1]][1] - 2, 0)
            tab[key] = (hi, n, n ** a.tau / (n ** a.tau).sum())
        while True:
            market = self.markets[rs.choice(len(self.markets), p=self.w)]
            have = [s for s in self.scales if (market, s) in mem.s]
            scale = have[rs.integers(len(have))]
            ser = mem.s[market, scale]
            hi, n, p = tab[market, scale]
            Lc, Lh, tmin = WINDOWS[scale]
            if (a.lam_x > 0 or a.peers > 0) and rs.random() < a.xs_prob:
                # One bar time across stocks, so the batch can score a cross-sectional rank.
                i = rs.choice(len(ser.codes), p=p)
                t = int(ser.M[int(ser.start[i] + 2 + rs.integers(n[i])) - 1])
                ends = [(j, ser.upto(j, t)) for j in range(len(ser.codes))]
                ends = [(j, e) for j, e in ends if e - 2 > ser.start[j] and e + Lh <= hi[j] and int(ser.M[e - 1]) == t]
                if len(ends) >= 16:
                    pick = rs.permutation(len(ends))[:a.batch]
                    rows = [mem.window(market, scale, ends[k][0], ends[k][1], Lc, Lh, mode=a.mode, mult=a.pyr_mult) for k in pick]
                    rank = a.lam_x > 0 and (a.peers == 0 or rs.random() < 0.5)
                    yield collate(rows), market, scale, Lh, 2 if rank else 1
                    continue
            rows = []
            for i in rs.choice(len(ser.codes), a.batch, p=p):
                end = int(ser.start[i] + 2 + recent_draw(rs, n[i], a.recent_hl))
                r = mem.window(market, scale, i, end, Lc, Lh, mode=a.mode, mult=a.pyr_mult)
                L = r["Lc"]
                if rs.random() < a.trunc_prob:
                    keep = int(rs.integers(min(tmin, L), L + 1))
                    r["x"][:L - keep] = 0
                    r["pad_mask"][:L - keep] = False
                    for k in ("clock", "sess", "yday", "dow", "month", "event"):
                        r[k][:L - keep] = 0
                rows.append(r)
            yield collate(rows), market, scale, Lh, 0


def recent_draw(rs, n, hl):
    """A bar offset in [0, n): uniform when hl is 0, else weighted by a half-life of hl bars back from the cut."""
    if not hl:
        return rs.integers(n)
    lam = np.log(2) / hl
    age = -np.log(1 - rs.random() * (1 - np.exp(-lam * n))) / lam
    return int(min(n - 1, max(0, n - 1 - int(age))))


def lr_at(a, s):
    if s < a.warmup:
        return a.lr * (s + 1) / a.warmup
    d0 = a.steps * (1 - a.decay_frac)
    if s < d0:
        return a.lr
    return a.min_lr + (a.lr - a.min_lr) * 0.5 * (1 + math.cos(math.pi * (s - d0) / (a.steps - d0)))


def flow_loss(model, b, Lh, market, scale, a, dev, xs=0):
    x = b["x"]
    B, L, _ = x.shape
    Lc = L - Lh
    x0 = x[:, Lc:]
    cond = {k: b[k] for k in FEATS}
    if a.pix:
        sig = vol_unit(x[:, :Lc], b["pad_mask"][:, :Lc], b["unit"], a.sig_hl)
        fut, x0 = x0, pix_target(x0, sig, b["unit"], a.pix, a.pix_z)
        if a.lam_y > 0:
            y = pix_y(candle_path(decode(fut.float(), b["unit"][:, None]))[3], b["unit"][:, 1] if a.y_raw else sig, a.pix_z)
            x0 = torch.cat([x0, y[..., None]], -1)
    drop = torch.ones(B, device=dev) if xs else torch.rand(B, device=dev)
    z_inst = drop < a.inst_drop
    cond["inst"] = cond["inst"].masked_fill(z_inst, 0)
    z_all = (drop >= a.inst_drop) & (drop < a.inst_drop + a.all_drop)
    for k in ("market", "sector", "inst", "scale"):
        cond[k] = cond[k].masked_fill(z_all, 0)
    ctx = x[:, :Lc].clone()
    z_hist = torch.ones(B, device=dev) < a.hist_drop if xs else torch.rand(B, device=dev) < a.hist_drop
    if z_hist.any():
        nh = make_nohist_cond({k: v[z_hist] for k, v in cond.items()}, Lc)
        for k in nh:
            cond[k] = cond[k].clone()
            cond[k][z_hist] = nh[k]
        ctx[z_hist] = 0
    if a.t_dist == "uniform":
        t = torch.rand(B, device=dev)
    else:
        t = torch.sigmoid(torch.randn(B, device=dev) * 1.0 + float(a.t_dist))
    if a.t1_prob > 0:  # t=1 trains the one-step conditional mean directly.
        t = torch.where(torch.rand(B, device=dev) < a.t1_prob, torch.ones_like(t), t)
    if xs == 2:
        t = torch.ones(B, device=dev)
    if a.pix_cls:
        # The same network as a per-pixel classifier: no noise, no flow, the baseline for the canvas.
        R = a.pix
        v = model(model.join(ctx, torch.zeros_like(x0)), torch.ones(B, device=dev), cond, Lh).float()
        loss = F.cross_entropy(v[..., :R].flatten(0, 1), x0[..., :R].argmax(-1).flatten()) + F.binary_cross_entropy_with_logits(v[..., R:], (x0[..., R:] + 1) / 2)
        aux = torch.zeros((), device=dev)
        if xs == 2:
            p = pix_expect(pix_probs(v, R), sig, R, a.pix_z)[:, -1]
            q = candle_path(decode(fut.float(), b["unit"][:, None]))[3][:, -1].argsort().argsort().float()
            aux = -F.cosine_similarity(p - p.mean(), q - q.mean(), dim=0)
        return loss + a.lam_x * aux, aux
    eps = torch.randn_like(x0)
    if a.src_a > 0:
        # The source holds real futures of other windows, so a flow that learns nothing still draws the climatology.
        bank = model.bank.get(Lh)
        src = {k: b[k] for k in FEATS}
        rat = src_ratio(x[:, :Lc], src, x0.shape[-1]) if a.src_vol else 1.0
        q = src_key(model, x[:, :Lc], src) if a.src_key else None
        mk = src["market"].expand(B) if a.src_mk else None
        if bank is not None and len(bank) >= 2 * B:
            idx = src_pick(model, Lh, q, mk, B, 1)[:, 0]
            eps = a.src_a * bank[idx] * rat + (1 - a.src_a ** 2) ** 0.5 * eps
        else:
            eps = a.src_a * x0[torch.randperm(B, device=dev)] + (1 - a.src_a ** 2) ** 0.5 * eps
        keep = (x0.detach().float() / rat)
        model.bank[Lh] = (keep if bank is None else torch.cat([bank, keep]))[-a.src_n:]
        if q is not None:
            model.bank_key[Lh] = (q if bank is None else torch.cat([model.bank_key[Lh], q]))[-a.src_n:]
        if mk is not None:
            model.bank_mk[Lh] = (mk if bank is None else torch.cat([model.bank_mk[Lh], mk]))[-a.src_n:]
    tv = t[:, None, None]
    z = (1 - tv) * x0 + tv * eps
    v = model(model.join(ctx, z), t, cond, Lh)
    loss = F.mse_loss(v, eps - x0)
    if a.pix and a.lam_y > 0:
        loss = F.mse_loss(v[..., :-1], (eps - x0)[..., :-1]) + a.lam_y * F.mse_loss(v[..., -1], (eps - x0)[..., -1])
    aux = torch.zeros((), device=dev)
    if xs == 2 and a.pix and a.lam_y > 0:
        # Rank through the numeric channel alone; the canvas stays a calibration target.
        p = (eps - v)[:, -1, -1]
        if not (a.xs_neut or a.y_raw):
            p = p * sig
        q = (x0[:, -1, -1] if a.xs_neut else candle_path(decode(fut.float(), b["unit"][:, None]))[3][:, -1]).argsort().argsort().float()
        aux = -F.cosine_similarity(p - p.mean(), q - q.mean(), dim=0)
        return loss + a.lam_x * aux, aux
    if xs == 2 and a.pix:
        # The expected return of the canvas is differentiable too.
        p = pix_expect(eps - v, sig, a.pix, a.pix_z)[:, -1]
        q = candle_path(decode(fut.float(), b["unit"][:, None]))[3][:, -1].argsort().argsort().float()
        aux = -F.cosine_similarity(p - p.mean(), q - q.mean(), dim=0)
        return loss + a.lam_x * aux, aux
    if xs == 2:
        # The one-step mean is differentiable, so its cross-sectional rank can be trained.
        sc = scale_tensor(STATS, market, scale, dev)
        p = decode(eps - v, sc[None, None])
        q = decode(x0, sc[None, None])
        p, q = (p[..., 0] + p[..., 1]).sum(1), (q[..., 0] + q[..., 1]).sum(1)
        q = q.argsort().argsort().float()
        aux = -F.cosine_similarity(p - p.mean(), q - q.mean(), dim=0)
        return loss + a.lam_x * aux, aux
    if a.lam_m > 0:
        sc = scale_tensor(STATS, market, scale, dev)
        xhat = z - tv * v
        o, hi, lo, c = candle_path(decode(xhat, sc[None, None]))
        o0, hi0, lo0, c0 = candle_path(decode(x0, sc[None, None]))
        unit = sc[1]
        for w in [2 ** k for k in range(1, int(math.log2(Lh)) + 1)]:
            p = measure(o, hi, lo, c, w)
            q = measure(o0, hi0, lo0, c0, w)
            den = unit * math.sqrt(w)
            for u, y in zip(p, q):
                aux = aux + F.smooth_l1_loss(u / den, y / den)
        loss = loss + a.lam_m * aux
    return loss, aux


def src_ratio(ctx, cond, D):
    """Recent volatility over the market's own unit, for the price features of a numeric future."""
    r = (vol_unit(ctx, cond["pad_mask"][:, :ctx.shape[1]], cond["unit"]) / cond["unit"][:, 1]).clamp(0.2, 5)
    r = r[:, None, None].expand(-1, 1, D).clone()
    r[..., 4:] = 1
    return r


@torch.no_grad()
def src_key(model, ctx, cond):
    """Retrieval key of a window: its chart seen by the EMA encoder, or return and volatility summaries."""
    net = model.__dict__.get("key_net", model)
    mask = cond["pad_mask"][:, :ctx.shape[1]]
    if model.src_key == 1:
        img, _ = render(ctx[:, :, :5], mask, cond["unit"], net.H)
        k = net.enc(img.to(net.axis.weight.dtype)).mean(1)
    else:
        raw = decode(ctx.float(), cond["unit"][:, None])
        r = (raw[..., 0] + raw[..., 1]) * mask
        sig = vol_unit(ctx, mask, cond["unit"])
        k = torch.stack([r[:, -n:].sum(1) / (sig * n ** 0.5) for n in (5, 20, 60)] + [torch.log(sig / cond["unit"][:, 1])], 1)
    return F.normalize(k.float(), dim=1)


def src_pick(model, Lh, q, mk, Q, n, gen=None):
    """n bank rows per query: drawn from its nearest windows when keyed, within its market when asked."""
    dev = model.bank[Lh].device
    N = len(model.bank[Lh])
    m = min(model.src_m, N)
    top = []
    for j in range(0, Q, 256):
        J = min(Q - j, 256)
        s = q[j:j + 256] @ model.bank_key[Lh].T if q is not None else torch.rand(J, N, device=dev, generator=gen)
        if mk is not None:
            s = s - 1e3 * (model.bank_mk[Lh][None] != mk[j:j + 256, None])
        top.append(s.topk(m, 1)[1])
    return torch.cat(top).gather(1, torch.randint(m, (Q, n), device=dev, generator=gen))


@torch.no_grad()
def sample(model, cond, ctx, Lh, K, S, dev, gen, canvas=False, zero=False, temp=1.0):
    """Euler integration from KiT's sampler. Also returns the one-step conditional mean."""
    B = ctx.shape[0]
    # Each chunk holds distinct stocks in a fresh order, so peer blocks see real neighbours.
    gen_cpu = torch.Generator().manual_seed(int(torch.randint(1 << 30, (1,), generator=gen, device=dev)))
    order = torch.cat([torch.randperm(B, generator=gen_cpu) for _ in range(K)]).to(dev)
    model.peer_on = len(model.peers) > 0
    c2 = {k: (v[order] if v.ndim else v) for k, v in cond.items()}
    ctx2 = ctx[order]
    z = torch.randn(B * K, Lh, model.cfg.in_features, generator=gen, device=dev) * temp  # sampling temperature (1 = trained prior)
    if model.cls or zero:
        z = torch.zeros_like(z)
    elif getattr(model, "bank", None):
        bank = model.bank[Lh]
        if getattr(model, "src_key", 0) or getattr(model, "src_mk", 0):
            q = src_key(model, ctx, cond) if model.src_key else None
            mk = cond["market"].expand(B) if model.src_mk else None
            idx = src_pick(model, Lh, q, mk, B, K, gen)[order, torch.arange(B * K, device=dev).div(B, rounding_mode="floor")]
        else:
            idx = torch.randint(len(bank), (B * K,), generator=gen, device=dev)
        rat = src_ratio(ctx2, c2, z.shape[-1]) if getattr(model, "src_vol", 0) else 1.0
        z = model.src_a * bank[idx] * rat + (1 - model.src_a ** 2) ** 0.5 * z
    first = None
    for i in range(1 if model.cls else S):
        t = 1 - i / S
        v = torch.cat([model(model.join(ctx2[j:j + 128], z[j:j + 128]), torch.full((len(z[j:j + 128]),), t, device=dev),
                             {k: u[j:j + 128] for k, u in c2.items()}, Lh) for j in range(0, B * K, 128)]).float()
        if model.cls:
            z = first = pix_probs(v, model.pix)
            break
        if i == 0:
            first = z - v
        z = z - v / S
    if model.pix and not canvas:
        sig = vol_unit(ctx2, c2["pad_mask"][:, :ctx.shape[1]], c2["unit"], getattr(model, "sig_hl", 0))
        z, first = (pix_numeric(u, c2["unit"][:, 1] if model.yraw else sig, c2["unit"], model.pix, model.zmax) for u in (z, first))
    # Back to [stock, draw]: position p holds stock order[p] in draw p // B.
    back = (order + B * torch.arange(B * K, device=dev).div(B, rounding_mode="floor")).argsort()
    z, first = z[back].view(K, B, Lh, -1).transpose(0, 1), first[back].view(K, B, Lh, -1).transpose(0, 1)
    return z, first


def attach_bank(model, ck, dev):
    if "bank" in ck:
        bank = ck["bank"] if isinstance(ck["bank"], dict) else {ck["bank"].shape[1]: ck["bank"]}
        model.bank, model.src_a = {k: v.float().to(dev) for k, v in bank.items()}, ck["args"]["src_a"]
        for k in ("src_vol", "src_key", "src_mk", "src_m"):
            setattr(model, k, ck["args"].get(k, 0))
        model.bank_key = {k: v.float().to(dev) for k, v in ck.get("bank_key", {}).items()}
        model.bank_mk = {k: v.to(dev) for k, v in ck.get("bank_mk", {}).items()}


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def rk(a):
    return np.argsort(np.argsort(a)).astype(np.float64)


@torch.no_grad()
def validate(model, mem, a, dev):
    """Per market and scale: anchors at session closes over the validation window."""
    out = {}
    gen = torch.Generator(dev).manual_seed(0)
    model.eval()
    for market in a.val_markets.split(","):
        for scale in a.scales.split(","):
            if (market, scale) not in mem.s:
                continue
            ser = mem.s[market, scale]
            Lc, Lh, _ = WINDOWS[scale]
            sc = scale_tensor(STATS, market, scale, dev)
            tdays = mem.tdays[market]
            hd = int(np.ceil(Lh / bars_day(market, scale)))
            span = tdays[(tdays >= np.datetime64(a.val_lo)) & (tdays <= np.datetime64(a.val_hi))]
            anchors = span[:-hd][::hd]
            sel = anchors[:: max(1, len(anchors) // a.val_anchors)][:a.val_anchors]
            ic, ric, vic, m_ic, m_ric = [], [], [], [], []
            close = {"ashare": 900, "cnq": 900, "us": 960, "fnspid": 960, "crypto": 1440}[market]
            limit = to_minutes(a.val_hi) + 1440 + 1
            for d in sel:
                t = to_minutes(str(d)) + close
                rows, acts = [], []
                for i in np.random.default_rng(0).permutation(len(ser.codes))[:a.val_inst]:
                    end = ser.upto(i, t)
                    if end <= ser.start[i] + 1 or end + Lh > ser.end[i] or abs(int(ser.M[end - 1]) - t) > 1 or ser.M[end + Lh - 1] > limit:
                        continue
                    rows.append(mem.window(market, scale, i, end, Lc, Lh, mode=a.mode, mult=a.pyr_mult, fut=False))
                    p = np.asarray(ser.O[end - 1:end + Lh], np.float64)
                    acts.append(np.diff(p))
                if len(rows) < 8:
                    continue
                b = {k: v.to(dev) for k, v in collate(rows).items()}
                Lc2 = rows[0]["Lc"]
                cond = {k: b[k] for k in FEATS}
                paths, first = sample(model, cond, b["x"][:, :Lc2], Lh, a.val_k, a.val_steps, dev, gen)
                raw = decode(paths, sc[None, None, None])
                bar = (raw[..., 0] + raw[..., 1]).float().cpu().numpy()
                mean = decode(first, sc[None, None, None])
                mean = (mean[..., 0] + mean[..., 1]).float().cpu().numpy()
                act = np.stack(acts)
                y = act.sum(-1)
                pr = bar.sum(-1).mean(1)
                ic.append(corr(pr, y))
                ric.append(corr(rk(pr), rk(y)))
                m = mean.sum(-1).mean(1)
                m_ic.append(corr(m, y))
                m_ric.append(corr(rk(m), rk(y)))
                vic.append(corr(rk(bar.std(-1, ddof=1).mean(1)), rk(act.std(-1, ddof=1))))
            out[f"{market}/{scale}"] = dict(ret_ic=float(np.nanmean(ic)), ret_rankic=float(np.nanmean(ric)), vol_rankic=float(np.nanmean(vic)),
                                            mean_ic=float(np.nanmean(m_ic)), mean_rankic=float(np.nanmean(m_ric)), n=len(ic))
    keys = ("ret_ic", "ret_rankic", "vol_rankic", "mean_ic", "mean_rankic")
    out["ALL"] = {k: float(np.nanmean([v[k] for n, v in out.items()])) for k in keys}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--mode", default="kit", choices=["kit", "pyr", "pyr250"])
    p.add_argument("--size", default="30m")
    p.add_argument("--n_layers", type=int, default=0)
    p.add_argument("--d_model", type=int, default=0)
    p.add_argument("--warm", default="")
    p.add_argument("--market_w", default="us:0.5,ashare:0.25,crypto:0.25")
    p.add_argument("--scales", default=",".join(SCALES))
    p.add_argument("--train_end", default="2024-12-31")
    p.add_argument("--val_lo", default="2025-01-01")
    p.add_argument("--val_hi", default="2025-12-31")
    p.add_argument("--val_markets", default="us,ashare,crypto")
    p.add_argument("--val_anchors", type=int, default=3)
    p.add_argument("--val_inst", type=int, default=200)
    p.add_argument("--val_k", type=int, default=4)
    p.add_argument("--val_steps", type=int, default=6)
    p.add_argument("--val_every", type=int, default=20000)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--decay_frac", type=float, default=0.2)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--tau", type=float, default=0.9)
    p.add_argument("--trunc_prob", type=float, default=0.3)
    p.add_argument("--inst_drop", type=float, default=0.1)
    p.add_argument("--all_drop", type=float, default=0.05)
    p.add_argument("--hist_drop", type=float, default=0.1)
    p.add_argument("--t_dist", default="uniform")
    p.add_argument("--t1_prob", type=float, default=0.0)
    p.add_argument("--chart", type=int, default=0, help="Chart image height; 0 keeps numeric history")
    p.add_argument("--pix", type=int, default=0, help="Rows of the future canvas; 0 predicts numeric candles")
    p.add_argument("--pix_z", type=float, default=4.0, help="Half height of the canvas in sig * sqrt(h)")
    p.add_argument("--pix_cls", type=int, default=0, help="Train the canvas as a per-pixel classifier")
    p.add_argument("--src_a", type=float, default=0.0, help="Share of real futures in the flow's source; 0 is pure noise")
    p.add_argument("--src_n", type=int, default=8192, help="Futures kept in the source bank")
    p.add_argument("--ctx_norm", type=int, default=0, help="Rescale numeric history per window: 1 by volatility, 2 by range")
    p.add_argument("--fuse", type=int, default=0, help="Feed the chart and rescaled numeric history together")
    p.add_argument("--enc_base", type=int, default=64, help="Channels of the chart encoder")
    p.add_argument("--vlm", default="", help="Frozen open VLM vision tower that also reads the chart (siglip2)")
    p.add_argument("--recent_hl", type=float, default=0.0, help="Half-life in bars, back from the training cut, of how often a window is drawn; 0 is uniform")
    p.add_argument("--sig_hl", type=float, default=0.0, help="Half-life in bars of the canvas volatility unit; 0 weighs the window evenly")
    p.add_argument("--level", type=int, default=0, help="Add the history chart redrawn in the future canvas's price frame")
    p.add_argument("--src_vol", type=int, default=0, help="Rescale a numeric source to the window's recent volatility")
    p.add_argument("--src_key", type=int, default=0, help="Draw the source from analogs: 1 by chart embedding, 2 by return summaries")
    p.add_argument("--src_mk", type=int, default=0, help="Draw the source from the window's own market")
    p.add_argument("--src_m", type=int, default=64, help="Nearest windows a keyed draw chooses among")
    p.add_argument("--lam_y", type=float, default=0.0, help="Weight of a numeric return channel beside the canvas; 0 drops it")
    p.add_argument("--y_raw", type=int, default=0, help="Numeric channel in market units; the default is recent volatility")
    p.add_argument("--xs_neut", type=int, default=0, help="Rank returns in units of recent volatility, not raw returns")
    p.add_argument("--lam_x", type=float, default=0.0)
    p.add_argument("--peers", type=int, default=0)
    p.add_argument("--xs_prob", type=float, default=0.25)
    p.add_argument("--lam_m", type=float, default=0.0)
    p.add_argument("--pyr_mult", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--win", default="", help="Override the 1d window as Lc/Lh/trunc_min")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    assert not (a.src_vol and a.pix), "A canvas is already in units of recent volatility"
    assert a.src_key != 1 or a.chart, "A chart key needs the chart encoder"
    if a.win:
        WINDOWS["1d"] = tuple(int(x) for x in a.win.split("/"))
    dev = "cuda"
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    make = lambda: build_model(a.size, a.n_layers or None, a.d_model or None, like=a.warm or None, peers=a.peers, chart=a.chart, pix=a.pix, zmax=a.pix_z, cls=bool(a.pix_cls), sub=a.lam_y > 0, yraw=bool(a.y_raw), norm=a.ctx_norm, fuse=a.fuse, base=a.enc_base, level=a.level, vlm=a.vlm, sig_hl=a.sig_hl)[0]
    model = make()
    model.bank, model.bank_key, model.bank_mk = {}, {}, {}
    model.src_key, model.src_mk, model.src_m = a.src_key, a.src_mk, a.src_m
    if a.warm:
        print("warm", load_kit_weights(model, a.warm), flush=True)
    model.to(dev)
    print(a.tag, "params", sum(p.numel() for p in model.parameters()) / 1e6, flush=True)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    rest = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=a.wd), dict(params=rest, weight_decay=0)], lr=a.lr, betas=(0.9, 0.95), eps=1e-15, fused=True)
    ema = make().to(dev)
    extra = lambda: dict(bank={k: v.half().cpu() for k, v in model.bank.items()}, bank_key={k: v.half().cpu() for k, v in model.bank_key.items()},
                         bank_mk={k: v.cpu() for k, v in model.bank_mk.items()}) if a.src_a > 0 else {}
    ema.load_state_dict(model.state_dict())
    model.__dict__["key_net"] = ema
    ema.requires_grad_(False)
    ds = Windows(a, a.seed)
    dl = DataLoader(ds, batch_size=None, num_workers=a.workers, prefetch_factor=4, persistent_workers=True)
    mem = Mem(a.val_markets.split(","), a.scales.split(","))
    if a.smoke:
        a.steps, a.val_every, a.val_anchors = 30, 30, 1
    os.makedirs("mf_runs", exist_ok=True)
    best, hist, t0 = -9, [], time.time()
    it = iter(dl)
    model.train()
    for s in range(a.steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.accum):
            b, market, scale, Lh, xs = next(it)
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            model.peer_on = xs > 0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, aux = flow_loss(model, b, Lh, market, scale, a, dev, xs)
            assert torch.isfinite(loss), (market, scale)
            (loss / a.accum).backward()
            tot += float(loss.detach()) / a.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr_at(a, s)
        opt.step()
        d = min(a.ema, (s + 1) / (s + 10))
        with torch.no_grad():
            for e, m in zip(ema.parameters(), model.parameters()):
                e.lerp_(m, 1 - d)
        if s % 100 == 0:
            print(f"step {s} loss {tot:.4f} aux {float(aux.detach()):.4f} {scale} {(s + 1) / (time.time() - t0):.2f} it/s", flush=True)
        if (s + 1) % a.val_every == 0 or s + 1 == a.steps:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if a.src_a > 0:
                    ema.bank, ema.src_a, ema.bank_key, ema.bank_mk = model.bank, a.src_a, model.bank_key, model.bank_mk
                    ema.src_vol, ema.src_key, ema.src_mk, ema.src_m = a.src_vol, a.src_key, a.src_mk, a.src_m
                r = validate(ema, mem, a, dev)
            r["step"] = s + 1
            hist.append(r)
            print(json.dumps(r["ALL"]), flush=True)
            score = max(r["ALL"]["ret_rankic"], r["ALL"]["mean_rankic"])
            if score > best:
                best = score
                torch.save(dict(ema=ema.state_dict(), args=vars(a), step=s + 1, **extra()), f"mf_runs/{a.tag}.pt")
            if s + 1 == a.steps:
                torch.save(dict(ema=ema.state_dict(), args=vars(a), step=s + 1, **extra()), f"mf_runs/{a.tag}_last.pt")
            model.train()
    save_result(a.tag, dict(args=vars(a), hist=hist, best=best, sec=time.time() - t0))


if __name__ == "__main__":
    main()
