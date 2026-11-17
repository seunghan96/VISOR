"""Classic single-task forecasters (DLinear, PatchTST, iTransformer, TimeMixer) as extra baselines.

Model code is the Time-Series-Library version shipped in MM-TSFlib (MMTSFLIB_ROOT), with its common default settings and no tuning.
Each model is trained from scratch with an MSE loss on the selector's own training windows and early-stopped on its
validation windows. It forecasts one point path, written in the same format as the TSFM outputs so the paper tables
read it unchanged:
  FNSPID   -> px_tsfm_out/fxt_<model>.npz   paths [10000, 1, 10] log return over the last close, rows of px_rows/fnspid_te23
  Time-MMD -> px_tsfm_out/tmfxt_<model>.npz paths [1111, 1, 12] 0.05 * z change over the last value, rows of tmmd_te
`python classic_ts.py --data fnspid|tmmd --model DLinear`
"""
import argparse
import glob
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visor import paths  # noqa: E402
sys.path.insert(0, paths.MMTSFLIB)
# SelfAttention_Family imports reformer_pytorch for LSH attention, which none of these four models uses.
sys.modules.setdefault("reformer_pytorch", SimpleNamespace(LSHSelfAttention=None))
MEM = paths.FNSPID


def build(name, L, H):
    import importlib
    cfg = SimpleNamespace(task_name="long_term_forecast", seq_len=L, label_len=0, pred_len=H, enc_in=1, dec_in=1, c_out=1,
                          d_model=64, d_ff=128, e_layers=2, n_heads=4, factor=1, dropout=0.1, activation="gelu",
                          embed="timeF", freq="d", moving_avg=25, output_attention=False, use_norm=1,
                          channel_independence=1, decomp_method="moving_avg", top_k=5, down_sampling_layers=1,
                          down_sampling_window=2, down_sampling_method="avg", num_class=1)
    return importlib.import_module(f"models.{name}").Model(cfg)


def fnspid_split(pat, nmax, rng):
    fs = sorted(glob.glob(f"sel_dump/fsel_fnspid_{pat}_p*.npz"))
    D = [np.load(f, allow_pickle=True) for f in fs]
    code = np.concatenate([d["code"] for d in D]); day = np.concatenate([d["day"] for d in D])
    sig = np.concatenate([d["sig"] for d in D]).astype(np.float64); y = np.concatenate([d["y"][:, :, 2] for d in D]).astype(np.float64)
    if nmax and len(code) > nmax:
        s = np.sort(rng.choice(len(code), nmax, replace=False))
        code, day, sig, y = code[s], day[s], sig[s], y[s]
    raw = {}
    X, keep = [], []
    for i, (c, d) in enumerate(zip(code, day)):
        if c not in raw:
            g = pd.read_parquet(f"{MEM}/raw/{c}.parquet")
            raw[c] = (g.datetime.values.astype("datetime64[D]"), g["close"].to_numpy(np.float64))
        t, v = raw[c]
        j = np.searchsorted(t, np.datetime64(d), side="right")
        if j >= 90:
            X.append(np.log(v[j - 90:j] / v[j - 1])); keep.append(i)
    keep = np.array(keep)
    Y = y[keep] * sig[keep, None] * np.sqrt(np.arange(1, 11))
    return np.array(X), Y


def load(a):
    rng = np.random.default_rng(0)
    if a.data == "fnspid":
        Xtr, Ytr = fnspid_split("tr_g17", a.ntrain, rng)
        Xva, Yva = fnspid_split("va_g17", 20000, rng)
        r = np.load("px_rows/fnspid_te23.npz")
        Xte = np.log(r["hist"][:, :, 3] / r["l0"][:, None])
        return [(None, Xtr, Ytr, Xva, Yva, Xte, np.ones(len(Xte), bool))], 100.0   # inputs in percent
    pre = "fintexts" if a.data == "fintexts" else "tmmd"
    side = np.load(f"sel_dump/{pre}_side.npz", allow_pickle=True)
    H = {(c, d): h for c, d, h in zip(side["code"], side["day"], side["hist"])}
    S = {}
    for s in ("tr", "va", "te"):
        z = np.load(f"sel_dump/{pre}_{s}.npz", allow_pickle=True)
        hz = np.sqrt(np.arange(1, z["y"].shape[1] + 1))
        X = np.stack([H[(c, d)] - H[(c, d)][-1] for c, d in zip(z["code"], z["day"])])
        Y = z["y"][:, :, 2] * hz * z["sig"][:, None] / 0.05
        S[s] = (z["code"], X, Y)
    if a.data == "fintexts":   # one model over all tickers, as the generator
        return [(None, S["tr"][1], S["tr"][2], S["va"][1], S["va"][2], S["te"][1], np.ones(len(S["te"][1]), bool))], 1.0
    out = []
    for ds in np.unique(S["te"][0]):
        m = {s: S[s][0] == ds for s in S}
        out.append((ds, S["tr"][1][m["tr"]], S["tr"][2][m["tr"]], S["va"][1][m["va"]], S["va"][2][m["va"]], S["te"][1][m["te"]], m["te"]))
    return out, 1.0


def fit(a, Xtr, Ytr, Xva, Yva, Xte, scale, dev):
    torch.manual_seed(0)
    L, H = Xtr.shape[1], Ytr.shape[1]
    net = build(a.model, L, H).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    T = lambda x: torch.tensor(x * scale, dtype=torch.float32, device=dev)[..., None]
    xtr, ytr, xva, yva = T(Xtr), T(Ytr), T(Xva), T(Yva)
    pred = lambda x: torch.cat([net(x[j:j + 4096], None, None, None)[..., 0] for j in range(0, len(x), 4096)])
    best, state, bad = 1e18, None, 0
    bs = min(a.bs, max(8, len(xtr) // 8))
    for ep in range(a.epochs):
        net.train()
        for idx in torch.randperm(len(xtr), device=dev).split(bs):
            loss = ((net(xtr[idx], None, None, None) - ytr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = ((pred(xva) - yva[..., 0]) ** 2).mean().item()
        if v < best:
            best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        return pred(T(Xte)).cpu().numpy() / scale, best, ep + 1


def main():
    q = argparse.ArgumentParser()
    q.add_argument("--data", required=True, choices=("fnspid", "tmmd", "fintexts"))
    q.add_argument("--model", required=True, choices=("DLinear", "PatchTST", "iTransformer", "TimeMixer"))
    q.add_argument("--ntrain", type=int, default=200000)
    q.add_argument("--epochs", type=int, default=30)
    q.add_argument("--patience", type=int, default=3)
    q.add_argument("--bs", type=int, default=256)
    q.add_argument("--lr", type=float, default=1e-3)
    q.add_argument("--tag", default="", help="Write results/<tag>.json when done")
    a = q.parse_args()
    t0 = time.time()
    parts, scale = load(a)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = np.zeros((len(parts[0][6]), parts[0][2].shape[1]))
    for ds, Xtr, Ytr, Xva, Yva, Xte, m in parts:
        pp, v, ep = fit(a, Xtr, Ytr, Xva, Yva, Xte, scale, dev)
        print(a.model, ds or a.data, "train", len(Xtr), "val mse", round(v, 4), "epochs", ep, flush=True)
        p[m] = pp
    if a.data == "fnspid":
        out = f"px_tsfm_out/fxt_{a.model.lower()}.npz"; paths = p[:, None]
    elif a.data == "fintexts":
        out = f"px_tsfm_out/ftxt_{a.model.lower()}.npz"; paths = 0.05 * p[:, None]
    else:
        out = f"px_tsfm_out/tmfxt_{a.model.lower()}.npz"; paths = 0.05 * p[:, None]
    np.savez(out, paths=paths.astype(np.float32), kind="point")
    print("wrote", out, paths.shape, f"{time.time() - t0:.0f}s")
    if a.tag:
        import json
        json.dump({"out": out, "model": a.model, "data": a.data}, open(f"results/{a.tag}.json", "w"))


if __name__ == "__main__":
    main()
