"""CFA ("Rethinking Multimodal Fusion for Time Series: Text Modalities Need Constrained Fusion", arXiv 2603.22372,
the Controlled Fusion Adapter) as a TS+text baseline, on exactly the windows of classic_ts.py / mm_ts.py.

Follows the released code (run.py, exp/exp_long_term_forecasting_text_integrated.py, scripts/run_timemmd.sh) in our
pipeline. The release is not shipped here: its backbones with the CFA adapter (models/*.py, layers/*) are imported
from CFA_ROOT via sys.path, and the training loop follows the release.
  prompt   = MM-TSFlib's "<|start_prompt|Make predictions about the future based on the following information: {text}...>"
  emb      = frozen BERT-base input (word) embeddings of the prompt (llm_model BERT, use_fullmodel 0, max_length 1024)
  c        = avg over tokens (pads included) of MLP(768 -> 96 -> text_emb = pred_len, ReLU, dropout 0.3)
  CFA      = h + W_up ReLU(LayerNorm(W_down c)), bottleneck D / r with r = cfa_reduction = 8, added to the output of
             every encoder layer (text_injection_mode "cfa"; DLinear: on its seasonal / trend outputs, D = channels)
  out      = backbone output (no prior_y mixing in the release)
Released script settings (scripts/run_timemmd.sh, run.py defaults otherwise): features S, label_len = seq_len / 2,
batch 32, 10 epochs, early stopping patience 5 on the val MSE, constant lr (adjust_learning_rate commented out),
seed 2021, d_model 512 / d_ff 2048 / 2 encoder + 1 decoder layers / 8 heads. Base lr: backbone (incl. the adapters)
1e-4, text MLP 1e-2 (mlp_proj 1e-3 is built but never used in the forward pass). The release evaluates each setting
over the script's ten lr multipliers {0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100} on all learning rates and keeps the
run with the lowest validation error (paper Sec. 5, "the model with the lowest validation error is selected"); we
do the same on our val split. best_settings.csv / best_setting.pkl of the release are the lr picks for its own windows
(seq_len 8/36/96), not ours, so they are not reused. As upstream the checkpoint keeps only the backbone (with its
adapters), so the text MLP stays at its last-epoch weights. The text encoder is BERT (the release's run.py default and
one of its four encoders; local weights, no external API). Time marks: none (x_mark None), as mm_ts.py / tats_ts.py.
Encoder-decoder backbones get x_dec = [last label_len inputs, zeros] as TSLib.
Backbones: from the release's run_timemmd.sh list, the ones overlapping our other baselines: PatchTST, iTransformer,
DLinear, Transformer, Autoformer.
Data / per-row text / prior handling: mm_ts.load (the selector's text of each row; Time-MMD in absolute z).
Outputs px_tsfm_out/{fxt,ftxt,tmfxt}_cfa_<model>.npz in classic_ts.py's format. Each lr run is cached in
results/cfa_parts/ so a restarted job resumes.
`python cfa_ts.py --data fnspid|fintexts|tmmd --model PatchTST [--tag T]`
"""
import argparse
import importlib
import json
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401

import mm_ts
from mm_ts import BERT, EXP, MLP, TextIds

from visor import paths  # noqa: E402  (mm_ts put the repository root on sys.path)

CFA = paths.CFA
LR_MULTS = (0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)   # scripts/run_timemmd.sh
DEC = ("Transformer", "Autoformer")


def build(name, L, H):
    """CFA backbone with run.py defaults (features S) and the CFA adapter."""
    if sys.path[0] != CFA:
        sys.path.insert(0, CFA)
    cfg = SimpleNamespace(task_name="long_term_forecast", seq_len=L, label_len=L // 2, pred_len=H, enc_in=1, dec_in=1, c_out=1,
                          d_model=512, d_ff=2048, e_layers=2, d_layers=1, n_heads=8, factor=1, dropout=0.1, activation="gelu",
                          embed="timeF", freq="h", moving_avg=25, output_attention=False, use_norm=1, distil=True,
                          channel_independence=1, decomp_method="moving_avg", top_k=5, num_kernels=6, num_class=1,
                          down_sampling_layers=0, down_sampling_window=1, down_sampling_method=None, seg_len=48,
                          text_emb=H, text_injection_mode="cfa", cfa_reduction=8, features="S")
    mod = importlib.import_module(f"models.{name}")
    assert mod.__file__.startswith(CFA), mod.__file__
    return mod.Model(cfg)


def fit(a, name, sp, scale, dev, emb, tok, mult):
    for f in (random.seed, np.random.seed, torch.manual_seed):
        f(a.seed)
    Xtr, Ytr = sp["tr"][0], sp["tr"][1]
    L, H = Xtr.shape[1], Ytr.shape[1]
    net = build(a.model, L, H).float().to(dev)
    mlp = MLP([768, 768 // 8, H], dropout_rate=0.3).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr * mult)
    opt2 = torch.optim.Adam(mlp.parameters(), lr=a.lr2 * mult)
    T = lambda x: torch.tensor(np.asarray(x) * scale, dtype=torch.float32, device=dev)[..., None]
    data = {s: (T(sp[s][0]), T(sp[s][1]), TextIds(tok, sp[s][3])) for s in sp}

    def forward(s, idx):
        x, y, ti = data[s]
        it = torch.as_tensor(idx, device=dev)
        c = mlp(emb(ti.batch(idx, dev))).mean(1)                    # [B, text_emb], avg pooling (pads included)
        xx = x[it]
        xd = None
        if a.model in DEC:
            xd = torch.cat([xx[:, -(L // 2):], torch.zeros(len(xx), H, 1, device=dev)], 1)
        return net(xx, None, xd, None, text_context=c)[:, -H:, :], y[it]

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
        print(f"{name} lr x{mult} epoch {ep + 1} train {np.mean(tl):.4f} val {v:.4f}", flush=True)
        if best is None or v < best:   # nan val never becomes best
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}   # backbone + adapters, as upstream
        else:
            bad += 1
            if bad >= a.patience:
                break
    if best_state is None or not np.isfinite(best):
        return None, float("inf"), ep + 1
    net.load_state_dict(best_state)
    pte, _ = predict("te")
    return pte, best, ep + 1


def main():
    q = argparse.ArgumentParser()
    q.add_argument("--data", required=True, choices=("fnspid", "tmmd", "fintexts"))
    q.add_argument("--model", required=True, choices=("PatchTST", "iTransformer", "DLinear", "Transformer", "Autoformer"))
    q.add_argument("--ntrain", type=int, default=200000, help="FNSPID train windows (as classic_ts.py)")
    q.add_argument("--epochs", type=int, default=10)
    q.add_argument("--patience", type=int, default=5)
    q.add_argument("--bs", type=int, default=32)
    q.add_argument("--lr", type=float, default=1e-4)
    q.add_argument("--lr2", type=float, default=1e-2, help="text MLP lr (upstream learning_rate2)")
    q.add_argument("--lr_mults", default=",".join(map(str, LR_MULTS)), help="the release's lr multipliers, picked on val")
    q.add_argument("--seed", type=int, default=2021)
    q.add_argument("--smoke", type=int, default=0, help="Use only the first N rows of each split; writes nothing")
    q.add_argument("--only", default="", help="Time-MMD: run one dataset only (smoke tests)")
    q.add_argument("--tag", default="", help="Write results/<tag>.json when done")
    a = q.parse_args()
    sys.path.insert(0, EXP)
    t0 = time.time()
    parts = mm_ts.load(a)
    print("data loaded", f"{time.time() - t0:.0f}s", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import BertModel, BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained(BERT)
    lm = BertModel.from_pretrained(BERT)
    emb = lm.get_input_embeddings().to(dev)
    for p_ in emb.parameters():
        p_.requires_grad = False
    del lm
    mults = [float(m) for m in a.lr_mults.split(",")]
    cache = "results/cfa_parts"
    os.makedirs(cache, exist_ok=True)
    n_te = len(parts[0][2])
    H = parts[0][1]["tr"][1].shape[1]
    p = np.zeros((n_te, H))
    info = {}
    for ds, sp, m, scale in parts:
        nm = ds or a.data
        runs = {}
        for mult in mults:
            cf = f"{cache}/{a.data}_{nm}_{a.model}_lr{mult}.npz"
            if os.path.exists(cf) and not (a.smoke or a.only):
                z = np.load(cf)
                runs[mult] = (z["pte"] if z["ok"] else None, float(z["val"]), int(z["ep"]))
                continue
            runs[mult] = fit(a, nm, sp, scale, dev, emb, tok, mult)
            if not (a.smoke or a.only):
                pte, v, ep = runs[mult]
                np.savez(cf, pte=np.zeros((0, H)) if pte is None else pte, ok=pte is not None, val=v, ep=ep)
        mult = min(runs, key=lambda k: runs[k][1])
        pp, v, ep = runs[mult]
        assert pp is not None, (nm, "every lr diverged")
        if a.data == "tmmd":
            pp = pp - sp["te"][0][:, -1:]          # absolute z -> change over the last value
            yt = sp["te"][1] - sp["te"][0][:, -1:]
        else:
            yt = sp["te"][1]
        info[nm] = dict(train=len(sp["tr"][0]), lr_mult=mult, val_mse=v, epochs=ep, val_by_lr={str(k): r[1] for k, r in runs.items()},
                        test_mae=float(np.abs(pp - yt).mean()), test_mae_last_value=float(np.abs(yt).mean()))
        print("CFA", a.model, nm, info[nm], flush=True)
        p[m] = pp
    pre = {"fnspid": "fxt", "fintexts": "ftxt", "tmmd": "tmfxt"}[a.data]
    paths = (p if a.data == "fnspid" else 0.05 * p)[:, None]
    if a.smoke or a.only:
        print("smoke run, nothing written", paths.shape, f"{time.time() - t0:.0f}s")
        return
    out = f"px_tsfm_out/{pre}_cfa_{a.model.lower()}.npz"
    np.savez(out, paths=paths.astype(np.float32), kind="point")
    print("wrote", out, paths.shape, f"{time.time() - t0:.0f}s")
    if a.tag:
        json.dump({"out": out, "model": "CFA-" + a.model, "data": a.data, "args": vars(a), "fit": info}, open(f"results/{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
