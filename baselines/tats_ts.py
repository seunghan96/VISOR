"""TaTS ("Language in the Flow of Time", ICLR 2026; github.com/iDEA-iSAIL-Lab-UIUC/TaTS, MIT, commit a053503)
as a TS+text baseline, on exactly the windows of classic_ts.py / mm_ts.py.

Reimplements the released exp/exp_long_term_forecasting.py + data_provider/data_loader.py (Dataset_Custom) in our
pipeline; the repo itself is only read (its data/<D>.csv, TATS_DATA), never run. Its model files are the MM-TSFlib / TSLib ones
up to comments, so the backbone comes from MM-TSFlib (MMTSFLIB_ROOT) as in classic_ts.py.
  each lookback step t has its own text (missing -> "No information available"), tokenized by GPT-2 (max_length 256)
  e_t  = masked average of GPT-2 input (wte) embeddings over the tokens of text t (use_fullmodel=0)
  c_t  = Dropout(ReLU(Linear(ReLU(Linear(e_t, 96)), text_emb=12)))        (MLP, dropout 0.3)
  x'   = [x_t, c_t] as 1 + 12 channels -> backbone -> output channel 0
  out  = (1 - w) * out + w * prior_y,  w = prior_weight = 0.5
Released script settings (scripts/main_forecast.sh): iTransformer, GPT2, text_emb 12, prior_weight 0.5, 5 epochs,
patience 5, seed 2025; run.py defaults otherwise (lr 1e-4, MLP lr 1e-2, batch 32, d_model 512). The checkpoint keeps
only the backbone (the MLP stays at its last epoch), as upstream. Only iTransformer mixes channels; with the
channel-independent DLinear / PatchTST the text channels never reach channel 0 (the repo's own note), so only
iTransformer is run.
Per-step text, cut at each step's own date (no future text):
  FNSPID   the stock's cleaned news of that trading day, in selector.News's format (clean=1, 8 items, 2 summaries)
  FinTexTS the dataset's LLM summaries of that day (fintexts.text_of, the selector's text, cut at 1500 chars)
  Time-MMD TaTS's released per-row `fact` column (its data/<D>.csv, rows identical to MM-TSFlib's); prior_y from its
           prior_history_avg. FNSPID / FinTexTS have no prior: prior_y = 0 in their last-value-relative units.
Writes px_tsfm_out/{fxt,ftxt,tmfxt}_tats.npz in classic_ts.py's format.
`python tats_ts.py --data fnspid|fintexts|tmmd [--tag T]`
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import mm_ts
from mm_ts import EXP, MEM, RAW, build

from visor import paths  # noqa: E402  (mm_ts put the repository root on sys.path)

TATS = paths.TATS_DATA
TATS_NAME = {"Algriculture": "Agriculture", "Public_Health": "Health"}
NOINFO = "No information available"
DEC = ("Transformer", "Informer", "Autoformer")   # encoder-decoder backbones (channel-mixing, like iTransformer)
# Extra channel-mixing backbones (the released paper reports TaTS on many TSLib backbones); the
# channel-independent ones (DLinear, PatchTST) stay out since the text channels never reach channel 0.
GPT2 = "gpt2"   # = openai-community/gpt2


def day_text(code, g, skip, boiler):
    """selector.News.text (clean=1, k=8, ns=2) restricted to one trading day's items."""
    g = g.iloc[::-1]
    own = re.compile(rf"(\b|\(){re.escape(code)}(\b|\))")
    keep, seen = [], set()
    for j, t in enumerate(g.title.values):
        key = " ".join(re.findall(r"[a-z]+", t.lower())[:6])
        if own.search(t) and not skip.search(t) and key not in seen:
            keep.append(j); seen.add(key)
    g = g.iloc[keep][:8]
    if not len(g):
        return None
    out = []
    for j, (dd, t, s) in enumerate(zip(g.day.values, g.title.values, g.summary.values)):
        line = f"- [{str(dd)[5:10]}] {t}"
        if j < 2 and isinstance(s, str):
            s = boiler.sub("", s).strip()
            if s:
                line += f": {s[:300]}"
        out.append(line)
    return "\n".join(out)


def fnspid(a):
    from visor.selector import BOILER, SKIP, load as vload
    rng = np.random.default_rng(0)
    Xtr, Ytr, ctr, dtr = mm_ts.fnspid_split("tr_g17", a.ntrain, rng)
    Xva, Yva, cva, dva = mm_ts.fnspid_split("va_g17", 20000, rng)
    r = np.load("px_rows/fnspid_te23.npz")
    TE = vload("sel_dump/fsel_fnspid_te_g17_p*.npz")
    cte, dte = TE["code"][r["rows"]], TE["day"][r["rows"]]
    Xte = np.log(r["hist"][:, :, 3] / r["l0"][:, None]); Yte = r["fut"].astype(np.float64)
    S = {"tr": (Xtr, Ytr, ctr, dtr), "va": (Xva, Yva, cva, dva), "te": (Xte, Yte, cte, dte)}
    if a.smoke:
        S = {k: tuple(v[:a.smoke] for v in S[k]) for k in S}
    # Dates of the 90 lookback steps of every window (the raw daily bars, as classic_ts.py's inputs).
    raw, steps = {}, {}
    for s, (X, _, C, Dd) in S.items():
        st = []
        for c, d in zip(C, Dd):
            if c not in raw:
                raw[c] = pd.read_parquet(f"{MEM}/raw/{c}.parquet").datetime.values.astype("datetime64[D]")
            t = raw[c]
            j = np.searchsorted(t, np.datetime64(d), side="right")
            st.append(t[j - X.shape[1]:j])
        steps[s] = np.array(st)
    flat = {s: pd.DataFrame({"code": np.repeat(S[s][2], steps[s].shape[1]), "d": steps[s].ravel()}) for s in S}
    need = pd.concat(flat.values()).drop_duplicates()
    nw = pd.read_parquet(f"{MEM}/news.parquet", columns=["code", "day", "title", "summary"])
    nw["d"] = nw.day.values.astype("datetime64[D]")
    nw = nw.merge(need, on=["code", "d"], how="inner", sort=False)
    texts, ks = [NOINFO], []
    for (c, dd), g in nw.groupby(["code", "d"], sort=False):
        t = day_text(c, g, SKIP, BOILER)
        if t:
            ks.append((c, dd, len(texts))); texts.append(t)
    key = pd.DataFrame(ks, columns=["code", "d", "kid"])
    key["d"] = key.d.values.astype("datetime64[D]")
    print("fnspid per-day texts", len(texts) - 1, "of", len(need), "lookback stock-days", flush=True)
    sp = {}
    for s, (X, Y, C, Dd) in S.items():
        kid = flat[s].merge(key, on=["code", "d"], how="left", sort=False).kid.fillna(0).to_numpy(np.int64)
        sp[s] = (X, Y, np.zeros_like(Y), kid.reshape(steps[s].shape))
    return [(None, sp, np.ones(len(sp["te"][0]), bool), 100.0)], texts


def fintexts(a):
    from visor import fintexts as BF
    W = BF.windows()
    texts, key = [NOINFO], {}
    sp = {}
    for s in ("tr", "va", "te"):
        z = np.load(f"sel_dump/fintexts_{s}.npz", allow_pickle=True)
        w = W[s]
        assert len(w) == len(z["code"]) and all(r[0] == c and r[1] == d for r, c, d in zip(w, z["code"], z["day"])), s
        hz = np.sqrt(np.arange(1, z["y"].shape[1] + 1))
        Y = z["y"][:, :, 2].astype(np.float64) * hz * z["sig"][:, None] / 0.05
        X = np.array([r[2] - r[2][-1] for r in w])
        assert np.abs((np.array([r[3] for r in w]) - np.array([r[2][-1:] for r in w])) - Y).max() < 2e-2
        idx = []
        for tk, _, h, _, j, g in w:
            row = []
            for k in range(j - len(h) + 1, j + 1):
                if (tk, k) not in key:
                    t = BF.text_of(g.iloc[k]).strip()[:1500]
                    key[(tk, k)] = len(texts) if t else 0
                    if t:
                        texts.append(t)
                row.append(key[(tk, k)])
            idx.append(row)
        sp[s] = (X, Y, np.zeros_like(Y), np.array(idx, np.int64))
    if a.smoke:
        sp = {k: tuple(v[:a.smoke] for v in sp[k]) for k in sp}
    return [(None, sp, np.ones(len(sp["te"][0]), bool), 1.0)], texts


def tmmd(a):
    L, H = 24, 12
    texts, out, full = [NOINFO], [], {s: [] for s in ("tr", "va", "te")}
    for d in ("Algriculture", "Climate", "Economy", "Energy", "Public_Health", "Security", "SocialGood", "Traffic"):
        g = pd.read_csv(glob.glob(f"{RAW}/{d}/*.csv")[0])
        tt = pd.read_csv(f"{TATS}/{TATS_NAME.get(d, d)}.csv")
        assert len(g) == len(tt) and (g.date.values == tt.date.values).all() and np.allclose(g.OT.fillna(0), tt.OT.fillna(0)), d
        g["fact"], g["tats_prior"] = tt["fact"].values, tt["prior_history_avg"].values
        g["date"] = pd.to_datetime(g.date)
        g = g.sort_values("date").reset_index(drop=True)   # as flow_mlp.py (Economy ships newest first)
        ot = g.OT.ffill().bfill().to_numpy(np.float64)
        n = len(g); ntr, nte = int(n * 0.7), int(n * 0.2); nva = n - ntr - nte
        mu, sd = ot[:ntr].mean(), ot[:ntr].std()
        z, pz = (ot - mu) / sd, (g.tats_prior.to_numpy(np.float64) - mu) / sd
        base = len(texts)
        texts += [str(t) if not pd.isnull(t) else NOINFO for t in g.fact.values]
        for s, lo, hi in zip(("tr", "va", "te"), [0, ntr - L, n - nte - L], [ntr, ntr + nva, n]):
            for t0 in range(lo, hi - L - H + 1):
                full[s].append((d, g.date.iloc[t0 + L - 1].strftime("%Y-%m-%d"), z[t0:t0 + L], z[t0 + L:t0 + L + H],
                                pz[t0 + L:t0 + L + H], base + np.arange(t0, t0 + L)))
    for s in full:   # same rows as the dumps
        zd = np.load(f"sel_dump/tmmd_{s}.npz", allow_pickle=True)
        assert len(zd["code"]) == len(full[s]) and all(r[0] == c and r[1] == dd for r, c, dd in zip(full[s], zd["code"], zd["day"])), s
    for ds in sorted({r[0] for r in full["te"]}):
        if a.only and ds != a.only:
            continue
        sp = {s: tuple(np.array([r[k] for r in full[s] if r[0] == ds]) for k in (2, 3, 4, 5)) for s in full}
        m = np.array([r[0] == ds for r in full["te"]])
        out.append((ds, sp, m, 1.0))
    return out, texts


def embed_texts(texts, dev, bs=512):
    """Masked average of GPT-2 input embeddings per text (max_length 256), as TaTS Dataset_Custom."""
    from transformers import GPT2Model, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(GPT2)
    tok.pad_token = tok.eos_token
    wte = GPT2Model.from_pretrained(GPT2).get_input_embeddings().to(dev)
    out = torch.zeros(len(texts), wte.weight.shape[1], dtype=torch.float32)
    with torch.no_grad():
        for j in range(0, len(texts), bs):
            t = tok(texts[j:j + bs], return_tensors="pt", padding=True, truncation=True, max_length=256)
            e = wte(t["input_ids"].to(dev))
            m = t["attention_mask"].to(dev).unsqueeze(-1).float()
            out[j:j + bs] = ((e * m).sum(1) / m.sum(1).clamp(min=1)).cpu()
    return out


def fit(a, name, sp, scale, dev, E):
    for f in (random.seed, np.random.seed, torch.manual_seed):
        f(a.seed)
    Xtr, Ytr = sp["tr"][0], sp["tr"][1]
    L, H = Xtr.shape[1], Ytr.shape[1]
    net = build(a.model, L, H, enc_in=1 + a.text_emb).float().to(dev)
    mlp = nn.Sequential(nn.Linear(768, 96), nn.ReLU(), nn.Linear(96, a.text_emb), nn.ReLU(), nn.Dropout(0.3)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    opt2 = torch.optim.Adam(mlp.parameters(), lr=a.lr2)
    T = lambda x: torch.tensor(np.asarray(x) * scale, dtype=torch.float32, device=dev)[..., None]
    data = {s: (T(sp[s][0]), T(sp[s][1]), T(sp[s][2]), torch.as_tensor(sp[s][3])) for s in sp}

    def forward(s, idx):
        x, y, p, ti = data[s]
        it = torch.as_tensor(idx, device=dev)
        e = E[ti[torch.as_tensor(idx)]].to(dev, non_blocking=True).float()      # [B, L, 768]
        xx = torch.cat([x[it], mlp(e)], -1)
        xd = None
        if a.model in DEC:   # decoder input as TSLib: the last label_len steps, then zeros for the horizon
            xd = torch.cat([xx[:, -(L // 2):], torch.zeros(len(xx), H, xx.shape[-1], device=dev)], 1)
        out = net(xx, None, xd, None)[:, -H:, :1]
        return (1 - a.prior_weight) * out + a.prior_weight * p[it], y[it]

    def predict(s):
        net.eval(); mlp.eval()
        n = len(data[s][0])
        with torch.no_grad():
            outs, losses = [], []
            for j in range(0, n, a.bs):
                o, y = forward(s, np.arange(j, min(n, j + a.bs)))
                outs.append(o[..., 0]); losses.append(((o - y) ** 2).mean().item())
        net.train(); mlp.train()
        return torch.cat(outs).cpu().numpy() / scale, float(np.mean(losses))

    best, best_state, bad, ntr = None, None, 0, len(data["tr"][0])
    for ep in range(a.epochs):
        perm = np.random.permutation(ntr)
        tl = []
        for j in range(0, ntr - a.bs + 1, a.bs):          # drop_last=True as upstream's train loader
            o, y = forward("tr", perm[j:j + a.bs])
            loss = ((o - y) ** 2).mean()
            opt.zero_grad(); opt2.zero_grad(); loss.backward(); opt.step(); opt2.step()
            tl.append(loss.item())
        _, v = predict("va")
        print(f"{name} epoch {ep + 1} train {np.mean(tl):.4f} val {v:.4f}", flush=True)
        if best is None or v < best:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    net.load_state_dict(best_state)
    pte, _ = predict("te")
    return pte, best, ep + 1


def main():
    q = argparse.ArgumentParser()
    q.add_argument("--data", required=True, choices=("fnspid", "tmmd", "fintexts"))
    q.add_argument("--model", default="iTransformer", choices=("iTransformer", "Transformer", "Informer", "Autoformer", "Crossformer"))
    q.add_argument("--ntrain", type=int, default=200000, help="FNSPID train windows (as classic_ts.py)")
    q.add_argument("--epochs", type=int, default=5)
    q.add_argument("--patience", type=int, default=5)
    q.add_argument("--bs", type=int, default=32)
    q.add_argument("--lr", type=float, default=1e-4)
    q.add_argument("--lr2", type=float, default=1e-2)
    q.add_argument("--text_emb", type=int, default=12)
    q.add_argument("--prior_weight", type=float, default=0.5)
    q.add_argument("--seed", type=int, default=2025)
    q.add_argument("--smoke", type=int, default=0)
    q.add_argument("--only", default="")
    q.add_argument("--tag", default="")
    a = q.parse_args()
    t0 = time.time()
    parts, texts = {"fnspid": fnspid, "fintexts": fintexts, "tmmd": tmmd}[a.data](a)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E = embed_texts(texts, dev).half()
    print("data + text embeddings", len(texts), f"{time.time() - t0:.0f}s", flush=True)
    n_te, H = len(parts[0][2]), parts[0][1]["tr"][1].shape[1]
    p, info = np.zeros((n_te, H)), {}
    for ds, sp, m, scale in parts:
        pp, v, ep = fit(a, ds or a.data, sp, scale, dev, E)
        yt = sp["te"][1]
        if a.data == "tmmd":
            pp, yt = pp - sp["te"][0][:, -1:], yt - sp["te"][0][:, -1:]
        info[ds or a.data] = dict(train=len(sp["tr"][0]), val_mse=v, epochs=ep, test_mae=float(np.abs(pp - yt).mean()),
                                  test_mae_last_value=float(np.abs(yt).mean()))
        print("TaTS", ds or a.data, info[ds or a.data], flush=True)
        p[m] = pp
    paths = (p if a.data == "fnspid" else 0.05 * p)[:, None]
    if a.smoke or a.only:
        print("smoke run, nothing written", paths.shape, f"{time.time() - t0:.0f}s")
        return
    pre = {"fnspid": "fxt", "fintexts": "ftxt", "tmmd": "tmfxt"}[a.data]
    out = f"px_tsfm_out/{pre}_tats.npz" if a.model == "iTransformer" else f"px_tsfm_out/{pre}_tats_{a.model.lower()}.npz"
    np.savez(out, paths=paths.astype(np.float32), kind="point")
    print("wrote", out, paths.shape, f"{time.time() - t0:.0f}s")
    if a.tag:
        json.dump({"out": out, "model": "TaTS-" + a.model, "data": a.data, "args": vars(a), "fit": info}, open(f"results/{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
