"""Candidate pools at other sampling temperatures, and mixed-temperature pools, from a saved text-free flow.

Nothing is retrained: the saved flow (sel_dump/<src>_gen.pt of flow_mlp.py) is loaded and only its sampler changes
(initial flow noise scaled by temp). temp < 1 gives calmer, less spread paths, temp > 1 rougher ones. Rows and their
order equal the temp-1 dumps (temp 1 reproduces them exactly, checked when <prefix>_<dataset> exists).

  python -m visor.temperature tmmd <dataset> --src <flow_mlp out> --prefix tmmdC <temp...>
        -> sel_dump/<prefix>_T<temp>_<dataset>_{tr,va,te}.npz
  python -m visor.temperature fintexts <temp...>   -> sel_dump/fintexts_T<temp>_{tr,va,te}.npz
  python -m visor.temperature mix <first-half prefix> <second-half prefix> <out prefix>
        -> half of the K paths from each pool (same rows), e.g. the Economy pool of the paper:
           mix tmmdC_T0.5_Economy tmmdC_T1.15_Economy tmmdC_mixw_Economy
Prints and saves the val/test medoid z-MAE and the candidate-to-truth volatility ratio (results/gentemp_<name>.json).
"""
import json
import sys

import numpy as np
import torch

from visor import flow_mlp as B


def sampler(net, sd, H, nfe):
    def f(h, k, seed, temp, bs=4096):
        net.eval()
        g = torch.Generator().manual_seed(seed)
        out = []
        for i in range(0, len(h), bs):
            c = torch.tensor(h[i:i + bs], dtype=torch.float32).repeat_interleave(k, 0)
            x = torch.randn(len(c), H, generator=g) * sd * temp
            dd = torch.zeros(len(c), dtype=torch.long)
            with torch.no_grad():
                for j in range(nfe):
                    x = x + net(x, torch.full((len(c),), j / nfe), c, dd) / nfe
            out.append(x.numpy().reshape(-1, k, H))
        return np.concatenate(out)
    return f


def stats(draws, f, sig):
    zc, zy = draws * sig[:, None, None], f * sig[:, None]
    med = np.abs(zc[:, :, None] - zc[:, None]).mean(-1).sum(-1).argmin(1)
    n = np.arange(len(zy))
    vol = lambda p: np.abs(np.diff(p, axis=-1, prepend=0)).mean(-1)
    return {"medoid": float(np.abs(zc[n, med] - zy).mean()), "median": float(np.abs(np.median(zc, 1) - zy).mean()),
            "oracle": float(np.abs(zc - zy[:, None]).mean(-1).min(1).mean()),
            "volratio": float(np.median(np.median(vol(zc), 1) / np.maximum(vol(zy), 1e-6)))}


def dump(path, rows_day, rows_code, draws, f, sig, S, H):
    lsig = S * sig
    root = np.sqrt(np.arange(1, H + 1))
    dr = (draws * S * sig[:, None, None]) / (lsig[:, None, None] * root)
    y = (f * S * sig[:, None]) / (lsig[:, None] * root)
    np.savez(path, day=rows_day, code=rows_code, sig=lsig.astype(np.float32), y=np.repeat(y[..., None], 3, -1).astype(np.float16),
             draws=np.repeat(dr[..., None], 3, -1).astype(np.float16))


def tmmd(ds, temps, src, prefix="tmmdC"):
    a = json.load(open(f"results/{src}.json"))["args"]
    W = B.windows()
    i = list(B.DOM).index(ds)
    trh = [w[2] for w in W["tr"] if w[0] == ds] + ([w[2] for w in W["extra"] if w[0] == ds] if a["extra"] else [])
    trf = [w[3] for w in W["tr"] if w[0] == ds] + ([w[3] for w in W["extra"] if w[0] == ds] if a["extra"] else [])
    _, fx, _ = B.norm(np.array(trh), np.array(trf))
    sd = torch.tensor(np.clip(fx, -30, 30), dtype=torch.float32).std(0, keepdim=True).clamp(min=1.0)
    net = B.Flow(len(B.DOM), a["width"], a["drop"], a["depth"])
    net.load_state_dict(torch.load(f"sel_dump/{src}_gen.pt")[i])
    f_s = sampler(net, sd, B.H, a["nfe"])
    rep = {}
    for t in temps:
        for s in ("tr", "va", "te"):
            rows = [w for w in W[s] if w[0] == ds]
            h, f, sig = B.norm(np.array([w[2] for w in rows]), np.array([w[3] for w in rows]))
            # The original dump drew all domains with one generator seeded a.seed+1 per domain call.
            draws = f_s(h, B.K, a["seed"] + 1, t)
            day, code = np.array([r[1] for r in rows]), np.array([r[0] for r in rows])
            if t == 1.0 and __import__("os").path.exists(f"sel_dump/{prefix}_{ds}_{s}.npz"):
                ref = np.load(f"sel_dump/{prefix}_{ds}_{s}.npz")
                lsig = B.S * sig
                dr = (draws * B.S * sig[:, None, None]) / (lsig[:, None, None] * np.sqrt(np.arange(1, B.H + 1)))
                err = np.abs(dr.astype(np.float16).astype(float) - ref["draws"][..., 0].astype(float)).max()
                print(ds, s, "temp 1 vs saved dump max diff", err, flush=True)
                assert (ref["day"] == day).all() and err < 1e-2, err
            else:
                dump(f"sel_dump/{prefix}_T{t}_{ds}_{s}.npz", day, code, draws, f, sig, B.S, B.H)
            if s != "tr":
                rep.setdefault(str(t), {})[s] = stats(draws, f, sig)
        print(ds, "temp", t, rep[str(t)], flush=True)
    json.dump(rep, open(f"results/gentemp_tmmd_{ds}.json", "w"), indent=1)


def fintexts(temps):
    from visor import fintexts as F
    B.L, B.H = F.L, F.H
    a = json.load(open("results/fintexts.json"))["args"]
    W = F.windows()
    arr = {s: (np.array([w[2] for w in W[s]]), np.array([w[3] for w in W[s]])) for s in W}
    _, fx, _ = B.norm(*arr["tr"])
    sd = torch.tensor(np.clip(fx, -30, 30), dtype=torch.float32).std(0, keepdim=True).clamp(min=1.0)
    net = B.Flow(1, a["width"], a["drop"], a["depth"])
    net.load_state_dict(torch.load("sel_dump/fintexts_gen.pt"))
    f_s = sampler(net, sd, F.H, a["nfe"])
    rep = {}
    for t in temps:
        for s in ("tr", "va", "te"):
            h, f, sig = B.norm(*arr[s])
            draws = f_s(h, F.K, a["seed"] + 1, t)
            day, code = np.array([r[1] for r in W[s]]), np.array([r[0] for r in W[s]])
            if t == 1.0:
                ref = np.load(f"sel_dump/fintexts_{s}.npz")
                lsig = F.S * sig
                dr = (draws * F.S * sig[:, None, None]) / (lsig[:, None, None] * np.sqrt(np.arange(1, F.H + 1)))
                err = np.abs(dr.astype(np.float16).astype(float) - ref["draws"][..., 0].astype(float)).max()
                print("fintexts", s, "temp 1 vs saved dump max diff", err, flush=True)
                assert (ref["day"] == day).all() and err < 1e-2, err
            else:
                dump(f"sel_dump/fintexts_T{t}_{s}.npz", day, code, draws, f, sig, F.S, F.H)
            if s != "tr":
                rep.setdefault(str(t), {})[s] = stats(draws, f, sig)
        print("fintexts temp", t, rep[str(t)], flush=True)
    json.dump(rep, open("results/gentemp_fintexts.json", "w"), indent=1)


def mix(first, second, out):
    """Mixed pool: the first K/2 paths of pool `first` and the last K/2 of pool `second` (independent draws, same rows)."""
    for s in ("tr", "va", "te"):
        a, b = np.load(f"sel_dump/{second}_{s}.npz"), np.load(f"sel_dump/{first}_{s}.npz")
        assert (a["day"] == b["day"]).all() and (a["code"] == b["code"]).all()
        k = a["draws"].shape[1] // 2
        d = np.concatenate([b["draws"][:, :k], a["draws"][:, k:]], 1)
        np.savez(f"sel_dump/{out}_{s}.npz", day=a["day"], code=a["code"], sig=a["sig"], y=a["y"], draws=d)
        print(out, s, d.shape)


if __name__ == "__main__":
    if sys.argv[1] == "tmmd":
        import argparse
        q = argparse.ArgumentParser()
        q.add_argument("dataset")
        q.add_argument("temps", nargs="+", type=float)
        q.add_argument("--src", required=True, help="flow_mlp.py --out of the dataset's predictor (sel_dump/<src>_gen.pt)")
        q.add_argument("--prefix", default="tmmdC", help="prefix of the dataset's temperature-1 pool")
        a = q.parse_args(sys.argv[2:])
        tmmd(a.dataset, a.temps, a.src, a.prefix)
    elif sys.argv[1] == "mix":
        mix(*sys.argv[2:5])
    else:
        fintexts([float(x) for x in sys.argv[2:]])
