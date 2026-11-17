"""Chart-Native predictor design for FinTexTS and Time-MMD: a text-free stochastic predictor trained in image space.
(On these datasets the TS-to-Chart flow MLP has the lower validation error and is the predictor of VISOR; this module
reproduces the design comparison. On FNSPID the Chart-Native predictor is the KiT canvas model of kit_flow.py.)

Each row's close history is drawn as a grayscale line chart (64 rows high, c pixel columns per step, one y-scale fixed
from the history alone). A conditional flow-matching UNet learns to draw the future columns of the chart (inpainting:
the history columns are given clean, the model only generates the future part). Different noise gives different
futures. Each generated chart is read back to numbers column by column (ink-weighted mean row at each future step's
column), so the candidates are scored with the same metrics and fed to the same selector (selector.py).

  python -m visor.chart_native check <ds>                  round-trip error of render -> decode on true charts (CPU)
  python -m visor.chart_native train <ds> [--sample_after 1] train on the train split, early stop on the val flow loss
  python -m visor.chart_native sample <ds> <split> <i/n>    K=32 decoded futures for part i of n of a split's rows
  python -m visor.chart_native cands <ds>                   merge parts -> sel_dump/<base>_img.npz (CPU), candidate stats
Inputs from chart_native_data.py (imgp_data/<ds>.npz). Units: history/future = (value - last) / sig, as in the dumps.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from visor import chart_native_data as D

# Per dataset: history length, horizon, pixel columns per step, axis reach G. Image height is HIMG for all.
# G: the y-axis is linear on the core band (history range padded by PADF * range + PADA * sqrt(H)); with G > 1 the outer
# (G - 1) / G of each half of the image holds a log-compressed margin reaching exp(G - 1) core half-widths, so heavy-tailed
# futures are not clipped. G per dataset = the smallest of 1, 1.5, 2, 2.5, 3 whose clip rate of the true futures is
# <= 0.5% on both the train and the val split (never test; chart_native.py check reports the clip rates of all splits).
CFG = {"fnspid": (90, 10, 2, 1.5), "fintexts": (64, 3, 3, 1.5), "Algriculture": (24, 12, 4, 1.5), "Climate": (24, 12, 4, 1.0),
       "Economy": (24, 12, 4, 1.0), "Energy": (24, 12, 4, 2.0), "Public_Health": (24, 12, 4, 3.0), "Security": (24, 12, 4, 1.5),
       "SocialGood": (24, 12, 4, 2.0), "Traffic": (24, 12, 4, 1.0)}
HIMG, LINE_SD, K = 64, 1.0, 32
PADF, PADA = 0.25, 1.0


def geom(ds):
    L, H, c, G = CFG[ds]
    W = -(-(L + H) * c // 8) * 8
    return L, H, c, W


def axis(h, ds):
    """Centre m, core half-width s and reach G of the row's chart, from the history alone (shared by all candidates)."""
    H, G = CFG[ds][1], CFG[ds][3]
    lo, hi = h.amin(1), h.amax(1)
    pad = PADF * (hi - lo) + PADA * math.sqrt(H)
    return dict(m=(lo + hi) / 2, s=(hi - lo) / 2 + pad, G=G)


def to_row(v, ax):
    x = (v - ax["m"][:, None]) / ax["s"][:, None]
    g = torch.where(x.abs() <= 1, x, x.sign() * (1 + torch.log(x.abs().clamp(min=1))))
    return ((1 - g / ax["G"]) / 2 * (HIMG - 1)).clamp(0, HIMG - 1)


def from_row(r, ax):
    g = ax["G"] * (1 - 2 * r / (HIMG - 1))
    x = torch.where(g.abs() <= 1, g, g.sign() * torch.exp(g.abs() - 1))
    return ax["m"][:, None] + ax["s"][:, None] * x


def render(v, ax, n_cols, c, W):
    """Values v [B,T] at steps 0..T-1 (step i at pixel column i*c) -> ink [B,HIMG,W] in [0,1], a soft line whose
    vertical width grows with the local slope so steep moves stay connected; columns >= n_cols are blank."""
    B, T = v.shape
    x = torch.arange(W, device=v.device, dtype=v.dtype)
    u = (x / c).clamp(0, T - 1)
    i0 = u.floor().long().clamp(max=max(T - 2, 0))
    fr = (u - i0)[None]
    v0, v1 = v[:, i0], v[:, (i0 + 1).clamp(max=T - 1)]
    yrow = to_row(v0 * (1 - fr) + v1 * fr, ax)
    slope = (to_row(v1, ax) - to_row(v0, ax)) / c
    sd = torch.sqrt(LINE_SD ** 2 + (slope / 2) ** 2)
    r = torch.arange(HIMG, device=v.device, dtype=v.dtype)[None, :, None]
    ink = torch.exp(-(r - yrow[:, None]) ** 2 / (2 * sd[:, None] ** 2))
    return ink * (x < n_cols).to(v.dtype)[None, None]


def decode(ink, ax, L, H, c, win=6, floor=0.1):
    """ink [B,HIMG,W] -> values [B,H] at the future steps' columns: ink-weighted mean row in a window around the peak."""
    cols = (L + torch.arange(H, device=ink.device)) * c
    col = ink[:, :, cols]                                   # [B, HIMG, H]
    r = torch.arange(HIMG, device=ink.device, dtype=ink.dtype)[None, :, None]
    pk = col.argmax(1, keepdim=True).to(ink.dtype)
    w = (col - floor).clamp(min=0) * ((r - pk).abs() <= win)
    yrow = torch.where(w.sum(1) > 1e-6, (w * r).sum(1) / w.sum(1).clamp(min=1e-6), pk[:, 0])
    return from_row(yrow, ax)


def make_batch(h, f, ds, dev):
    """Clean target chart x1 in [-1,1], the conditioning chart (history only, future columns blank) and the future mask."""
    L, H, c, W = geom(ds)
    h = torch.as_tensor(h, dtype=torch.float32, device=dev)
    ax = axis(h, ds)
    cond = render(h, ax, L * c, c, W) * 2 - 1
    out = dict(cond=cond, ax=ax)
    if f is not None:
        f = torch.as_tensor(f, dtype=torch.float32, device=dev)
        out["x1"] = render(torch.cat([h, f], 1), ax, (L + H) * c, c, W) * 2 - 1
    m = torch.zeros(W, device=dev)
    m[L * c:(L + H) * c] = 1
    out["mask"] = m[None, None].expand(len(h), 1, HIMG, W)
    return out


# ---------------- model: a small conditional UNet ----------------

def temb(t, d):
    f = torch.exp(-math.log(10000) * torch.arange(d // 2, device=t.device) / (d // 2))
    a = t[:, None] * 1000 * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Res(nn.Module):
    def __init__(self, ci, co, td, drop):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, ci), nn.Conv2d(ci, co, 3, padding=1)
        self.t = nn.Linear(td, co)
        self.n2, self.c2 = nn.GroupNorm(8, co), nn.Conv2d(co, co, 3, padding=1)
        self.drop = nn.Dropout(drop)
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x))) + self.t(te)[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.n, self.qkv, self.o = nn.GroupNorm(8, ch), nn.Conv2d(ch, 3 * ch, 1), nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, Hh, Ww = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, 4, C // 4, Hh * Ww).transpose(-1, -2).unbind(1)
        o = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(B, C, Hh, Ww)
        return x + self.o(o)


class UNet(nn.Module):
    def __init__(self, ch=(32, 64, 96, 128), nres=1, drop=0.1):
        super().__init__()
        td = 4 * ch[0]
        self.td0 = ch[0]
        self.tm = nn.Sequential(nn.Linear(ch[0], td), nn.SiLU(), nn.Linear(td, td))
        self.inp = nn.Conv2d(2, ch[0], 3, padding=1)
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        skips, c = [ch[0]], ch[0]
        for i, co in enumerate(ch):
            for _ in range(nres):
                self.down.append(Res(c, co, td, drop)); c = co; skips.append(c)
            if i < len(ch) - 1:
                self.pool.append(nn.Conv2d(c, c, 3, stride=2, padding=1)); skips.append(c)
        self.mid1, self.mida, self.mid2 = Res(c, c, td, drop), Attn(c), Res(c, c, td, drop)
        self.up, self.ups = nn.ModuleList(), nn.ModuleList()
        for i, co in reversed(list(enumerate(ch))):
            for _ in range(nres + 1):
                self.up.append(Res(c + skips.pop(), co, td, drop)); c = co
            if i > 0:
                self.ups.append(nn.Conv2d(c, c, 3, padding=1))
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))
        self.nres, self.nlev = nres, len(ch)

    def forward(self, x, mask, t):
        te = self.tm(temb(t, self.td0))
        h = self.inp(torch.cat([x, mask], 1))
        hs, j = [h], 0
        for i in range(self.nlev):
            for _ in range(self.nres):
                h = self.down[j](h, te); j += 1; hs.append(h)
            if i < self.nlev - 1:
                h = self.pool[i](h); hs.append(h)
        h = self.mid2(self.mida(self.mid1(h, te)), te)
        j = 0
        for i in range(self.nlev):
            for _ in range(self.nres + 1):
                h = self.up[j](torch.cat([h, hs.pop()], 1), te); j += 1
            if i < self.nlev - 1:
                h = self.ups[i](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.out(h)


def velocity(net, xt, b, t):
    """The model sees the clean history columns and the noisy future columns only."""
    m = b["mask"]
    x = m * xt + (1 - m) * b["cond"]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=xt.is_cuda):
        return net(x, m, t).float()


def fm_loss(net, b, gen=None):
    x1 = b["x1"][:, None]
    x0 = torch.randn(x1.shape, device=x1.device, generator=gen)
    t = torch.rand(len(x1), device=x1.device, generator=gen)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
    b = dict(b, cond=b["cond"][:, None])
    v = velocity(net, xt, b, t)
    m = b["mask"]
    return (((v - (x1 - x0)) ** 2) * m).sum((1, 2, 3)) / m.sum((1, 2, 3))


@torch.no_grad()
def sample_rows(net, h, ds, k, seed, nfe=32, bs=1024, dev="cuda", keep_img=0):
    """K generated futures per row, decoded to numbers: [N, k, H] (sig units). keep_img: also return a few images."""
    L, H, c, W = geom(ds)
    g = torch.Generator(device=dev).manual_seed(seed)
    out, imgs = [], []
    hr = np.repeat(h, k, 0)
    for i in range(0, len(hr), bs):
        b = make_batch(hr[i:i + bs], None, ds, dev)
        b["cond"] = b["cond"][:, None]
        x = torch.randn((len(b["cond"]), 1, HIMG, W), device=dev, generator=g)
        for j in range(nfe):
            t = torch.full((len(x),), j / nfe, device=dev)
            x = x + velocity(net, x, b, t) / nfe
        m = b["mask"]
        img = (m * x + (1 - m) * b["cond"])[:, 0].clamp(-1, 1)
        out.append(decode((img + 1) / 2, b["ax"], L, H, c).cpu().numpy())
        if keep_img and len(imgs) * bs < keep_img:
            imgs.append(img[:keep_img].cpu().numpy())
    out = np.concatenate(out).reshape(len(h), k, H).astype(np.float32)
    return (out, np.concatenate(imgs)[:keep_img]) if keep_img else out


def cand_stats(fut, f):
    """Candidate quality in the selector's draw units (close / sqrt(h)): mean candidate MAE, medoid, oracle, spread, vol ratio."""
    H = f.shape[1]
    r = np.sqrt(np.arange(1, H + 1))
    dr, y = fut / r, f / r
    err = np.abs(dr - y[:, None]).mean(-1)
    d1 = np.abs(dr[:, :, None] - dr[:, None]).mean(-1)
    med = d1.sum(-1).argmin(1)
    vol = lambda p: np.abs(np.diff(p, axis=-1, prepend=0)).mean(-1)
    return {"n": int(len(f)), "cand_mae": float(err.mean()), "medoid_mae": float(err[np.arange(len(f)), med].mean()),
            "oracle_mae": float(err.min(1).mean()), "median_mae": float(np.abs(np.median(dr, 1) - y).mean()),
            "spread": float(d1.mean() * d1.shape[1] / max(d1.shape[1] - 1, 1)),
            "end_sd": float(fut[:, :, -1].std(1).mean()), "volratio": float(np.median(np.median(vol(fut), 1) / np.maximum(vol(f), 1e-6)))}


# ---------------- modes ----------------

def check(ds, n=3000):
    """Round trip on the true charts: render history + true future, decode, compare (sig units and draw units)."""
    z = np.load(f"imgp_data/{ds}.npz")
    L, H, c, W = geom(ds)
    out = {}
    for sp in ("va", "te"):
        idx = np.random.default_rng(0).choice(len(z[f"{sp}_h"]), min(n, len(z[f"{sp}_h"])), replace=False)
        h, f = z[f"{sp}_h"][idx], z[f"{sp}_f"][idx]
        b = make_batch(h, f, ds, "cpu")
        ax = b["ax"]
        dec = decode((b["x1"] + 1) / 2, ax, L, H, c).numpy()
        e = np.abs(dec - f)
        reach = (ax["s"] * (math.exp(ax["G"] - 1) if ax["G"] > 1 else 1)).numpy()[:, None]
        inr = np.abs(f - ax["m"].numpy()[:, None]) <= reach
        clip = 1 - inr.mean()
        out[sp] = {"mae_sig": float(e.mean()), "p99_sig": float(np.quantile(e, 0.99)), "max_sig": float(e.max()),
                   "mae_draw_units": float((e / np.sqrt(np.arange(1, H + 1))).mean()),
                   "truth_abs_draw_units": float(np.abs(f / np.sqrt(np.arange(1, H + 1))).mean()),
                   "mae_inrange_sig": float(e[inr].mean()), "clip_frac": float(clip),
                   "core_sig_per_row": float(np.median(2 * ax["s"].numpy() / ((HIMG - 1) / ax["G"])))}
    print(ds, json.dumps(out), flush=True)
    return out


def train(a):
    ds, dev = a.ds, a.dev
    torch.manual_seed(a.seed)
    z = np.load(f"imgp_data/{ds}.npz")
    th, tf, vh, vf = z["gtr_h"], z["gtr_f"], z["gva_h"], z["gva_f"]
    print(ds, "train windows", len(th), "val windows", len(vh), "geom", geom(ds), flush=True)
    arch = dict(ch=tuple(map(int, a.ch.split(","))), nres=a.nres, drop=a.drop)
    net = UNet(**arch).to(dev).to(memory_format=torch.channels_last)
    ema = UNet(**arch).to(dev); ema.load_state_dict(net.state_dict()); ema.eval().requires_grad_(False)
    print("params", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500))
    # Small val sets (Time-MMD) are repeated with fresh fixed noise so the early-stopping loss is not too noisy.
    rep = -(-1000 // len(vh))
    vh2, vf2 = np.tile(vh, (rep, 1)), np.tile(vf, (rep, 1))
    vb = [make_batch(vh2[i:i + 512], vf2[i:i + 512], ds, dev) for i in range(0, len(vh2), 512)]

    def val_loss(m):
        g = torch.Generator(device=dev).manual_seed(123)
        with torch.no_grad():
            return float(torch.cat([fm_loss(m, b, g) for b in vb]).mean())

    rng = np.random.default_rng(a.seed)
    best, best_state, bad, t0, log = 1e9, None, 0, time.time(), []
    for step in range(1, a.steps + 1):
        i = rng.integers(0, len(th), a.bs)
        loss = fm_loss(net, make_batch(th[i], tf[i], ds, dev)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        dec = min(a.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.lerp_(p, 1 - dec)
        if step % a.eval_every == 0 or step == a.steps:
            v = val_loss(ema)
            log.append((step, float(loss), v))
            print(f"step {step} loss {loss.item():.4f} val {v:.4f} {(time.time() - t0) / step:.3f}s/step", flush=True)
            if v < best:
                best, bad, best_state = v, 0, {k: t.detach().clone() for k, t in ema.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    print("early stop", flush=True)
                    break
    ema.load_state_dict(best_state)
    os.makedirs("imgp_runs", exist_ok=True)
    torch.save(dict(state=best_state, args=vars(a)), f"imgp_runs/{a.run_name or ds}.pt")
    # Quick val check: K=16 decoded draws on up to 300 val rows.
    n = min(300, len(vh))
    fut = sample_rows(ema, vh[:n], ds, 16, 7, a.nfe, dev=dev)
    res = dict(args=vars(a), best_val_loss=best, log=log, val_quick=cand_stats(fut, vf[:n]), minutes=(time.time() - t0) / 60)
    print("val quick", res["val_quick"], flush=True)
    if a.sample_after:
        for sp in ("tr", "va", "te"):
            res[f"sample_{sp}"] = sample(ds, sp, "0/1", ema, a.nfe, dev, a.max_rows, a.run_name)
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/{a.tag or 'imgp_gen_' + ds}.json", "w"), indent=1)


def load_net(ds, dev="cuda"):
    ck = torch.load(f"imgp_runs/{ds}.pt", map_location=dev)
    A = ck["args"]
    net = UNet(tuple(map(int, A["ch"].split(","))), A["nres"], A["drop"]).to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    return net, ck["args"]


def sample(ds, sp, part, net=None, nfe=32, dev="cuda", max_rows=0, run_name=""):
    i, n = map(int, part.split("/"))
    z = np.load(f"imgp_data/{ds}.npz")
    if net is None:
        net, args = load_net(ds, dev)
        nfe = args["nfe"]
    rows = np.array_split(np.arange(len(z[f"{sp}_h"])), n)[i]
    rows = rows[:max_rows] if max_rows else rows
    t0 = time.time()
    seed = 1000 + 100 * ("tr", "va", "te").index(sp) + i
    fut, imgs = sample_rows(net, z[f"{sp}_h"][rows], ds, K, seed, nfe, dev=dev, keep_img=64)
    os.makedirs("imgp_out", exist_ok=True)
    np.savez(f"imgp_out/{run_name or ds}_{sp}_p{i}of{n}.npz", idx=rows, fut=fut, imgs=imgs.astype(np.float16))
    st = cand_stats(fut, z[f"{sp}_f"][rows])
    st["minutes"] = (time.time() - t0) / 60
    print(ds, sp, part, st, flush=True)
    return st


def cands(ds):
    """Merge sampled parts and write the selector files: the base rows unchanged, draws = decoded image futures."""
    z = np.load(f"imgp_data/{ds}.npz")
    L, H, c, W = geom(ds)
    rep = {}
    for sp in ("tr", "va", "te"):
        fs = sorted(glob.glob(f"imgp_out/{ds}_{sp}_p*of*.npz"))
        n = int(fs[0].split("of")[-1][:-4])
        assert len(fs) == n, fs
        P = [np.load(f) for f in fs]
        idx = np.concatenate([p["idx"] for p in P])
        fut = np.concatenate([p["fut"] for p in P])[np.argsort(idx)]
        assert np.array_equal(np.sort(idx), np.arange(len(z[f"{sp}_rows"])))
        Bf = [np.load(f, allow_pickle=True) for f in D.base_files(ds, sp)]
        rows = z[f"{sp}_rows"]
        base = {k: np.concatenate([d[k] for d in Bf])[rows] for k in Bf[0].files}
        f_true = D.fut_of(base["y"])
        assert np.allclose(f_true, z[f"{sp}_f"], atol=1e-3)
        old = base["draws"][..., 2].astype(np.float32) * np.sqrt(np.arange(1, H + 1))
        r = np.sqrt(np.arange(1, H + 1))[None, None, :, None]
        base["draws"] = np.repeat(fut[..., None], 3, -1) / r
        base["draws"] = base["draws"].astype(np.float16)
        assert np.isfinite(base["draws"]).all()
        out = os.path.basename(D.base_files(ds, sp)[0])
        out = f"sel_dump/{out.split('_p')[0] if ds == 'fnspid' else out[:-4]}_img.npz"
        np.savez(out, **base)
        rep[sp] = {"file": out, "img": cand_stats(fut, f_true), "option1_same_rows": cand_stats(old, f_true)}
        print(ds, sp, json.dumps(rep[sp]), flush=True)
    json.dump(rep, open(f"results/imgp_cands_{ds}.json", "w"), indent=1)
    return rep


if __name__ == "__main__":
    for d in ("results", "imgp_runs", "imgp_out", "sel_dump"):
        os.makedirs(d, exist_ok=True)
    q = argparse.ArgumentParser()
    q.add_argument("mode", choices=("check", "train", "sample", "cands"))
    q.add_argument("ds")
    q.add_argument("split", nargs="?", default="te")
    q.add_argument("part", nargs="?", default="0/1")
    q.add_argument("--steps", type=int, default=40000)
    q.add_argument("--bs", type=int, default=128)
    q.add_argument("--lr", type=float, default=2e-4)
    q.add_argument("--drop", type=float, default=0.1)
    q.add_argument("--ch", default="32,64,96,128")
    q.add_argument("--nres", type=int, default=1)
    q.add_argument("--ema", type=float, default=0.999)
    q.add_argument("--eval_every", type=int, default=1000)
    q.add_argument("--patience", type=int, default=8)
    q.add_argument("--nfe", type=int, default=32)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--sample_after", type=int, default=0)
    q.add_argument("--tag", default="")
    q.add_argument("--dev", default="cuda")
    q.add_argument("--run_name", default="", help="smoke tests: write imgp_runs/<run_name>.pt and imgp_out/<run_name>_* instead of <ds>")
    q.add_argument("--max_rows", type=int, default=0, help="smoke tests only: sample at most this many rows")
    a = q.parse_args()
    if a.mode == "check":
        json.dump(check(a.ds), open(f"results/imgp_check_{a.ds}.json", "w"), indent=1)
    elif a.mode == "train":
        train(a)
    elif a.mode == "sample":
        st = sample(a.ds, a.split, a.part, dev=a.dev, max_rows=a.max_rows)
        tag = a.tag or f"imgp_samp_{a.ds}_{a.split}_p{a.part.replace('/', 'of')}"
        json.dump(st, open(f"results/{tag}.json", "w"), indent=1)
    else:
        cands(a.ds)
