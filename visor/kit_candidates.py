"""Candidate futures of the FNSPID Chart-Native predictor for the selector: K sampled canvases per row with the truth.

Saved per row (sel_dump/<tag>.npz): day, code, the history features, its mask and unit, the draws and the truth as
cumulative (high, low, close) in path units (sig * sqrt(h)), and sig. A sampled canvas is read back as the center of
the close bin and as a random point in the top and bottom marked bins for the high and the low (canvas_hlc).
--codes limits the stocks to a list, and --news_days keeps a row only if the stock has news on one of the last that
many trading days (the news file maps each article to the first close after it). Text never reaches the predictor.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from visor import paths
from visor.kit_flow import Mem, build_model, candle_path, collate, decode, horizon, scale_tensor, to_minutes, vol_unit
from visor.kit_train import FEATS, STATS, attach_bank, sample

CLOSE = {"ashare": 900, "us": 960, "fnspid": 960, "crypto": 1440}
Lc, Lh = 90, 10


def rows_on(mem, ser, market, keep, d):
    t0 = to_minutes(str(d)) + CLOSE[market]
    rows, codes, ends = [], [], []
    for c in keep:
        i = ser.pos[c]
        end = ser.upto(i, t0)
        if end - ser.start[i] < Lc or abs(int(ser.M[end - 1]) - t0) > 1 or end + Lh > ser.end[i]:
            continue
        rows.append(mem.window(market, "1d", i, end, Lc, Lh))
        codes.append(c)
    return rows, codes


def canvas_hlc(z, R, zmax, u, gen):
    """Sampled canvases [...,Lh,2R] to cumulative (high, low, close) [...,Lh,3] in log return, u the row unit."""
    edge = torch.linspace(-zmax, zmax, R + 1, device=z.device)
    width = 2 * zmax / R
    cl = z[..., :R].argmax(-1)
    lit = z[..., R:] > 0
    rows = torch.arange(R, device=z.device)
    top = torch.where(lit, rows, -1).amax(-1)
    bot = torch.where(lit, rows, R).amin(-1)
    top = torch.maximum(torch.where(top < 0, cl, top), cl)
    bot = torch.minimum(torch.where(bot >= R, cl, bot), cl)
    jit = lambda r: edge[r] + torch.rand(r.shape, device=z.device, generator=gen) * width
    hz, lz, cz = jit(top) * u, jit(bot) * u, (edge[cl] + width / 2) * u
    return torch.stack([torch.maximum(hz, cz), torch.minimum(lz, cz), cz], -1)


def main():
    q = argparse.ArgumentParser()
    q.add_argument("--ckpt", required=True)
    q.add_argument("--market", required=True)
    q.add_argument("--lo", required=True)
    q.add_argument("--hi", required=True)
    q.add_argument("--stride", type=int, default=10)
    q.add_argument("--max_inst", type=int, default=400)
    q.add_argument("--k", type=int, default=32)
    q.add_argument("--steps", type=int, default=32)
    q.add_argument("--part", default="0/1")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--codes", default="")
    q.add_argument("--news_days", type=int, default=0)
    q.add_argument("--news", default=f"{paths.FNSPID}/news.parquet")
    q.add_argument("--temp", type=float, default=1.0, help="Scale of the initial flow noise (sampling temperature)")
    q.add_argument("--tag", required=True)
    a = q.parse_args()
    dev = "cuda"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    t = ck["args"]
    R, zmax = t["pix"], t["pix_z"]
    model = build_model(t["size"], t["n_layers"] or None, t["d_model"] or None, peers=t.get("peers", 0), chart=t.get("chart", 0),
                        pix=R, zmax=zmax, fuse=t.get("fuse", 0), base=t.get("enc_base", 64), level=t.get("level", 0),
                        vlm=t.get("vlm", ""), sig_hl=t.get("sig_hl", 0.0))[0]
    model.load_state_dict(ck["ema"])
    attach_bank(model, ck, dev)
    model.to(dev).eval()
    mem = Mem([a.market], ["1d"])
    ser = mem.s[a.market, "1d"]
    sc = scale_tensor(STATS, a.market, "1d", dev)
    if a.codes:
        keep = [c for c in open(a.codes).read().split() if c in ser.pos]
    else:
        keep = list(np.random.default_rng(0).permutation(ser.codes)[:a.max_inst])
    td = mem.tdays[a.market]
    if a.news_days:
        nw = pd.read_parquet(a.news, columns=["code", "day"]).drop_duplicates()
        nw = nw[nw.code.isin(set(keep))]
        pos = np.searchsorted(td, nw.day.values.astype("datetime64[D]"))
        has = {}
        for c, p in zip(nw.code.values, pos):
            for j in range(p, min(p + a.news_days, len(td))):
                has.setdefault(td[j], set()).add(c)
    days = td[(td >= np.datetime64(a.lo)) & (td <= np.datetime64(a.hi))][::a.stride]
    i0, n0 = (int(x) for x in a.part.split("/"))
    days = days[i0::n0]
    h = horizon(Lh, dev)
    gen = torch.Generator(dev).manual_seed(a.seed)
    torch.manual_seed(a.seed)
    out = {k: [] for k in ("day", "code", "x", "mask", "unit", "sig", "y", "draws")}
    for d in days:
        rows, codes = rows_on(mem, ser, a.market, [c for c in keep if c in has.get(d, ())] if a.news_days else keep, d)
        if len(rows) < 8:
            continue
        b = {k: v.to(dev) for k, v in collate(rows).items()}
        ctx = b["x"][:, :Lc]
        m = b["pad_mask"][:, :Lc]
        sig = vol_unit(ctx, m, b["unit"])
        sig_m = vol_unit(ctx, m, b["unit"], model.sig_hl) if model.sig_hl else sig
        u = sig[:, None] * h
        _, hi_, lo_, c_ = candle_path(decode(b["x"][:, Lc:].float(), sc[None, None]))
        y = torch.stack([hi_, lo_, c_], -1) / u[..., None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, _ = sample(model, {k: b[k] for k in FEATS}, ctx, Lh, a.k, a.steps, dev, gen, canvas=True, temp=a.temp)
        dr = canvas_hlc(z.float(), R, zmax, sig_m[:, None, None] * h, gen) / u[:, None, :, None]
        N = len(rows)
        for k, v in (("day", np.full(N, str(d), dtype="U10")), ("code", np.array(codes)), ("x", ctx[..., :5].half()),
                     ("mask", m), ("unit", b["unit"].float()), ("sig", sig.float()), ("y", y.half()), ("draws", dr.half())):
            out[k].append(v.cpu().numpy() if torch.is_tensor(v) else v)
        print(d, N, flush=True)
    os.makedirs("sel_dump", exist_ok=True)
    np.savez(f"sel_dump/{a.tag}.npz", **{k: np.concatenate(v) for k, v in out.items()})
    os.makedirs("results", exist_ok=True)
    json.dump(dict(args=vars(a), rows=int(sum(len(v) for v in out["day"]))), open(f"results/{a.tag}.json", "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
