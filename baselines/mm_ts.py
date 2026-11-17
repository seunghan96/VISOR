"""MM-TSFlib (Time-MMD, NeurIPS 2024 D&B) text fusion on top of the classic backbones, as a TS+text baseline.

Reimplements upstream HEAD (github.com/AdityaLab/MM-TSFlib, commit e789ce7: exp/exp_long_term_forecasting.py, run.py,
data_provider/data_loader.py) in our pipeline, on exactly the windows of classic_ts.py:
  prompt  = "<|start_prompt|Make predictions about the future based on the following information: {text}<|<end_prompt>|>"
  emb     = frozen BERT-base input (word) embeddings of the prompt (use_fullmodel=0: the LM itself is not run)
  emb     = MLP(768 -> 96 -> pred_len, ReLU, dropout 0.3) per token, then avg pooling over the tokens (pads included)
  prompt_y = norm(emb) + prior_y          (norm: zero mean, unit var over the horizon)
  out     = (1 - w) * backbone(x) + w * prompt_y,  w = prompt_weight = 0.1 (the released script)
Backbone Adam lr 1e-4, MLP Adam lr 1e-2, batch 32, 10 epochs, early stopping patience 5 on the val MSE, constant lr,
seed 2024, backbone dims from run.py defaults. As upstream, the checkpoint keeps only the backbone, so the text MLP
stays at its last-epoch weights.
Text of a row = the text the selector (visor/selector.py) sees for that row, cut at the same window end:
  FNSPID   News(n_days=5, clean=1).text(code, day, 8, 2) (the frozen config), FinTexTS / Time-MMD TNews (side file).
prior_y: Time-MMD uses the CSV's prior_history_avg at the forecast rows, scaled by the train-part OT scaler, and works
in absolute z as upstream. FNSPID / FinTexTS have no such column: prior_y = 0 in their last-value-relative units
(a last-value prior).
Outputs px_tsfm_out/{fxt,ftxt,tmfxt}_mmtsf_<model>.npz in classic_ts.py's format.
`python mm_ts.py --data fnspid|fintexts|tmmd --model PatchTST [--tag T]`
"""
import argparse
import glob
import json
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visor import paths  # noqa: E402
sys.path.insert(0, paths.MMTSFLIB)
sys.modules.setdefault("reformer_pytorch", SimpleNamespace(LSHSelfAttention=None))
EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
MEM = paths.FNSPID
RAW = paths.TMMD_RAW
PROMPT = "<|start_prompt|Make predictions about the future based on the following information: {}<|<end_prompt>|>"
BERT = "google-bert/bert-base-uncased"


def build(name, L, H, enc_in=1):
    """TSLib backbone with MM-TSFlib run.py defaults (features S, enc_in 1; TaTS adds the text channels)."""
    import importlib
    cfg = SimpleNamespace(task_name="long_term_forecast", seq_len=L, label_len=L // 2, pred_len=H, enc_in=enc_in, dec_in=enc_in, c_out=1,
                          d_model=512, d_ff=2048, e_layers=2, d_layers=1, n_heads=8, factor=1, dropout=0.1, activation="gelu",
                          embed="timeF", freq="h", moving_avg=25, output_attention=False, use_norm=1, distil=True,
                          channel_independence=1, decomp_method="moving_avg", top_k=5, num_kernels=6, num_class=1,
                          down_sampling_layers=0, down_sampling_window=1, down_sampling_method=None, seg_len=48)
    if name == "TimeMixer":
        # run.py's defaults (0 layers, method None) crash TimeMixer upstream too; use classic_ts.py's setting.
        cfg.down_sampling_layers, cfg.down_sampling_window, cfg.down_sampling_method = 1, 2, "avg"
    return importlib.import_module(f"models.{name}").Model(cfg)


class MLP(nn.Module):
    """Upstream MLP: Linear layers, ReLU + dropout between them."""

    def __init__(self, sizes, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.dropout(F.relu(x))
        return x


def norm(e):
    e = e - e.mean(1, keepdim=True).detach()
    return e / torch.sqrt(torch.var(e, dim=1, keepdim=True, unbiased=False) + 1e-5)


# ---------------------------------------------------------------- data (windows identical to classic_ts.py)

def fnspid_split(pat, nmax, rng):
    """classic_ts.fnspid_split, also returning each window's code and day (same rng draws, same windows)."""
    fs = sorted(glob.glob(f"sel_dump/fsel_fnspid_{pat}_p*.npz"))
    D = [np.load(f, allow_pickle=True) for f in fs]
    code = np.concatenate([d["code"] for d in D]); day = np.concatenate([d["day"] for d in D])
    sig = np.concatenate([d["sig"] for d in D]).astype(np.float64); y = np.concatenate([d["y"][:, :, 2] for d in D]).astype(np.float64)
    if nmax and len(code) > nmax:
        s = np.sort(rng.choice(len(code), nmax, replace=False))
        code, day, sig, y = code[s], day[s], sig[s], y[s]
    raw, X, keep = {}, [], []
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
    return np.array(X), Y, code[keep], day[keep]


def tmmd_prior():
    """Per Time-MMD window (flow_mlp.windows order): scaled prior_history_avg at the H forecast rows, and the targets."""
    L, H = 24, 12
    out = {s: [] for s in ("tr", "va", "te")}
    for d in ("Algriculture", "Climate", "Economy", "Energy", "Public_Health", "Security", "SocialGood", "Traffic"):
        g = pd.read_csv(glob.glob(f"{RAW}/{d}/*.csv")[0])
        g["date"] = pd.to_datetime(g.date)
        g = g.sort_values("date").reset_index(drop=True)   # as flow_mlp.py
        ot = g.OT.ffill().bfill().to_numpy(np.float64)
        n = len(g); ntr, nte = int(n * 0.7), int(n * 0.2); nva = n - ntr - nte
        mu, sd = ot[:ntr].mean(), ot[:ntr].std()           # upstream StandardScaler on the train part of OT
        z, pz = (ot - mu) / sd, (g.prior_history_avg.to_numpy(np.float64) - mu) / sd
        for s, lo, hi in zip(("tr", "va", "te"), [0, ntr - L, n - nte - L], [ntr, ntr + nva, n]):
            for t0 in range(lo, hi - L - H + 1):
                out[s].append((d, g.date.iloc[t0 + L - 1].strftime("%Y-%m-%d"), z[t0:t0 + L], z[t0 + L:t0 + L + H], pz[t0 + L:t0 + L + H]))
    return out


def load(a):
    """List of (name, splits, test_mask, scale); splits[s] = (X, Y, prior, texts) in model units."""
    rng = np.random.default_rng(0)
    if a.data == "fnspid":
        from visor.selector import News, load as vload
        Xtr, Ytr, ctr, dtr = fnspid_split("tr_g17", a.ntrain, rng)
        Xva, Yva, cva, dva = fnspid_split("va_g17", 20000, rng)
        r = np.load("px_rows/fnspid_te23.npz")
        TE = vload("sel_dump/fsel_fnspid_te_g17_p*.npz")
        cte, dte = TE["code"][r["rows"]], TE["day"][r["rows"]]
        Xte = np.log(r["hist"][:, :, 3] / r["l0"][:, None]); Yte = r["fut"].astype(np.float64)
        td = pd.to_datetime(pd.read_csv(f"{MEM}/1d/trade_dates.csv").date.astype(str)).values.astype("datetime64[D]")
        news = News(set(ctr) | set(cva) | set(cte), td, 5, 1, 0)
        txt = lambda C, Dd: [news.text(c, d, 8, 2) for c, d in zip(C, Dd)]
        S = {"tr": (Xtr, Ytr, ctr, dtr), "va": (Xva, Yva, cva, dva), "te": (Xte, Yte, cte, dte)}
        if a.smoke:
            S = {k: tuple(v[:a.smoke] for v in S[k]) for k in S}
        sp = {k: (X, Y, np.zeros_like(Y), txt(C, Dd)) for k, (X, Y, C, Dd) in S.items()}
        return [(None, sp, np.ones(len(sp["te"][0]), bool), 100.0)]
    from visor.selector import TNews, SIDE
    pre = "fintexts" if a.data == "fintexts" else "tmmd"
    SIDE[0] = f"sel_dump/{pre}_side.npz"
    tn = TNews()
    side = np.load(SIDE[0], allow_pickle=True)
    Hh = {(c, d): h for c, d, h in zip(side["code"], side["day"], side["hist"])}
    S = {}
    for s in ("tr", "va", "te"):
        z = np.load(f"sel_dump/{pre}_{s}.npz", allow_pickle=True)
        hz = np.sqrt(np.arange(1, z["y"].shape[1] + 1))
        Y = z["y"][:, :, 2].astype(np.float64) * hz * z["sig"][:, None] / 0.05       # z change over the last value
        T = [tn.text(c, d) for c, d in zip(z["code"], z["day"])]
        if pre == "fintexts":
            X = np.stack([Hh[(c, d)] - Hh[(c, d)][-1] for c, d in zip(z["code"], z["day"])])
            S[s] = (z["code"], X, Y, np.zeros_like(Y), T)
        else:
            S[s] = (z["code"], z["day"], Y, T)
    if pre == "fintexts":
        sp = {k: S[k][1:] for k in S}
        if a.smoke:
            sp = {k: tuple(v[:a.smoke] for v in sp[k]) for k in sp}
        return [(None, sp, np.ones(len(sp["te"][0]), bool), 1.0)]
    # Time-MMD: absolute z inputs / targets and the CSV prior, as upstream; rows checked against the dumps.
    W = tmmd_prior()
    full = {}
    for s in S:
        code, day, Y, T = S[s]
        w = W[s]
        assert len(w) == len(code) and all(a_ == c and b_ == d for (a_, b_, *_), c, d in zip(w, code, day)), s
        X = np.array([r[2] for r in w]); Ya = np.array([r[3] for r in w]); P = np.array([r[4] for r in w])
        err = np.abs((Ya - X[:, -1:]) - Y).max()
        assert err < 2e-2, (s, err)   # dumps store y in float16
        full[s] = (code, X, Ya, P, T)
    out = []
    for ds in np.unique(full["te"][0]):
        m = {s: full[s][0] == ds for s in full}
        sp = {s: (full[s][1][m[s]], full[s][2][m[s]], full[s][3][m[s]], [t for t, k in zip(full[s][4], m[s]) if k]) for s in full}
        if a.only and ds != a.only:
            continue
        out.append((ds, sp, m["te"], 1.0))
    return out


# ---------------------------------------------------------------- model

class TextIds:
    """Token ids of each row's prompt (BERT tokenizer, truncation 1024 as upstream); batches pad to their longest row."""

    def __init__(self, tok, texts):
        enc = tok([PROMPT.format(t) for t in texts], truncation=True, max_length=1024)["input_ids"]
        self.ids = [np.asarray(e, np.int64) for e in enc]
        self.pad = tok.pad_token_id

    def batch(self, idx, dev):
        L = max(len(self.ids[i]) for i in idx)
        out = np.full((len(idx), L), self.pad, np.int64)
        for k, i in enumerate(idx):
            out[k, :len(self.ids[i])] = self.ids[i]
        return torch.from_numpy(out).to(dev)


def fit(a, name, sp, scale, dev, emb, tok):
    for f in (random.seed, np.random.seed, torch.manual_seed):
        f(a.seed)
    Xtr, Ytr, Ptr, Ttr = sp["tr"]
    L, H = Xtr.shape[1], Ytr.shape[1]
    net = build(a.model, L, H).float().to(dev)
    mlp = MLP([768, 768 // 8, H], dropout_rate=0.3).to(dev)        # text_emb = pred_len
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    opt2 = torch.optim.Adam(mlp.parameters(), lr=a.lr2)
    T = lambda x: torch.tensor(np.asarray(x) * scale, dtype=torch.float32, device=dev)[..., None]
    data = {s: (T(sp[s][0]), T(sp[s][1]), T(sp[s][2]), TextIds(tok, sp[s][3])) for s in sp}

    def forward(s, idx):
        x, y, p, ti = data[s]
        it = torch.as_tensor(idx, device=dev)
        e = mlp(emb(ti.batch(idx, dev)))                                    # [B, tokens, H]
        e = F.adaptive_avg_pool1d(e.transpose(1, 2), 1).squeeze(2).unsqueeze(-1)   # [B, H, 1]
        xd = None
        if a.model == "Transformer":   # decoder input as TSLib (and tats_ts.py): the last label_len steps, then zeros
            xd = torch.cat([x[it][:, -(L // 2):], torch.zeros(len(it), H, 1, device=dev)], 1)
        out = net(x[it], None, xd, None)[:, -H:, :]
        return (1 - a.prompt_weight) * out + a.prompt_weight * (norm(e) + p[it]), y[it]

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
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}   # backbone only, as upstream
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
    q.add_argument("--model", required=True, choices=("DLinear", "PatchTST", "iTransformer", "TimeMixer", "Transformer"))
    q.add_argument("--ntrain", type=int, default=200000, help="FNSPID train windows (as classic_ts.py)")
    q.add_argument("--epochs", type=int, default=10)
    q.add_argument("--patience", type=int, default=5)
    q.add_argument("--bs", type=int, default=32)
    q.add_argument("--lr", type=float, default=1e-4)
    q.add_argument("--lr2", type=float, default=1e-2, help="text MLP lr (upstream learning_rate2)")
    q.add_argument("--prompt_weight", type=float, default=0.1, help="released script week_health.sh (run.py default 0.01)")
    q.add_argument("--seed", type=int, default=2024)
    q.add_argument("--smoke", type=int, default=0, help="Use only the first N rows of each split; writes nothing to px_tsfm_out")
    q.add_argument("--only", default="", help="Time-MMD: run one dataset only (smoke tests)")
    q.add_argument("--tag", default="", help="Write results/<tag>.json when done")
    a = q.parse_args()
    sys.path.insert(0, EXP)
    t0 = time.time()
    parts = load(a)
    print("data loaded", f"{time.time() - t0:.0f}s", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import BertModel, BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained(BERT)
    if tok.eos_token:
        tok.pad_token = tok.eos_token
    lm = BertModel.from_pretrained(BERT)
    emb = lm.get_input_embeddings().to(dev)
    for p_ in emb.parameters():
        p_.requires_grad = False
    del lm
    n_te = len(parts[0][2])
    H = parts[0][1]["tr"][1].shape[1]
    p = np.zeros((n_te, H))
    info = {}
    for ds, sp, m, scale in parts:
        pp, v, ep = fit(a, ds or a.data, sp, scale, dev, emb, tok)
        if a.data == "tmmd":
            pp = pp - sp["te"][0][:, -1:]          # absolute z -> change over the last value
            yt = sp["te"][1] - sp["te"][0][:, -1:]
        else:
            yt = sp["te"][1]
        info[ds or a.data] = dict(train=len(sp["tr"][0]), val_mse=v, epochs=ep, test_mae=float(np.abs(pp - yt).mean()),
                                  test_mae_last_value=float(np.abs(yt).mean()))
        print(a.model, ds or a.data, info[ds or a.data], flush=True)
        p[m] = pp
    pre = {"fnspid": "fxt", "fintexts": "ftxt", "tmmd": "tmfxt"}[a.data]
    paths = (p if a.data == "fnspid" else 0.05 * p)[:, None]
    if a.smoke or a.only:
        print("smoke run, nothing written", paths.shape, f"{time.time() - t0:.0f}s")
        return
    out = f"px_tsfm_out/{pre}_mmtsf_{a.model.lower()}.npz"
    np.savez(out, paths=paths.astype(np.float32), kind="point")
    print("wrote", out, paths.shape, f"{time.time() - t0:.0f}s")
    if a.tag:
        json.dump({"out": out, "model": a.model, "data": a.data, "args": vars(a), "fit": info}, open(f"results/{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
