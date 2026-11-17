"""VISOR selector: a VLM reads the text and one grid image of all K candidate charts, and picks one candidate.

The predictor never sees text: it draws K candidate futures from the history alone (kit_candidates.py, flow_mlp.py).
Here every candidate is drawn after the recent history as a small line (or candle) chart, and the K charts are placed
as numbered panels of one grid image on a shared value axis, in a random order per window. Qwen2.5-VL reads one
sequence per window: the text available before the forecast, the grid image and one marker token per candidate. The
last hidden state at each marker and at the last prompt token goes through a two-layer order-free transformer head that
gives one score per candidate. With --obj reg the scores regress the negative shape-aware distance of each candidate to
the truth (target_dist, Eq. "target" of the paper) up to a constant per window. LoRA adapters train on the language
model and, with --lora_vision 1, on the vision tower. --aux adds the auxiliary chart-reading loss.

Evaluation-only options: --text none / oracle, --ctrl (text controls), --anon (hidden ticker), and a shuffled-text
control that is always scored when --text news.
"""
import argparse
import glob
import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore", category=FutureWarning)

from visor import paths
from visor.metrics import evaluate, features, objective, report

VLM = "Qwen/Qwen2.5-VL-3B-Instruct"
MEM = paths.FNSPID
W, H, NB = 448, 224, 40
BG, INK, UP, DN, SHADE, FUT = (255, 255, 255), (30, 30, 30), (38, 120, 90), (200, 60, 50), (236, 240, 246), (40, 90, 170)


def load(pats):
    fs = sorted(f for p in pats.split(",") for f in glob.glob(p))
    ds = [np.load(f) for f in fs]
    out = {k: np.concatenate([d[k] for d in ds]) for k in ("day", "code", "sig", "y", "draws")}
    out["mk"] = np.zeros(len(out["day"]), int)
    print(len(fs), "files", len(out["day"]), "rows", flush=True)
    return out


class Prices:
    """Raw daily candles per stock, for the history part of each chart."""

    def __init__(self, codes, nb=NB):
        self.d, self.nb = {}, nb
        for c in set(codes):
            g = pd.read_parquet(f"{MEM}/raw/{c}.parquet")
            self.d[c] = (g.datetime.values.astype("datetime64[D]"), g[["open", "high", "low", "close"]].to_numpy(np.float64))

    def hist(self, code, day):
        t, v = self.d[code]
        i = np.searchsorted(t, np.datetime64(day), side="right")
        h = v[max(0, i - self.nb):i]
        return h / h[-1, 3]


BOILER = re.compile(r"(?i)(free stock analysis report|click to get this free report|to read this article on zacks(\.com click here)?|\bcom click here"
                    r"|image source: zacks investment research)[^.]*\.?")
SKIP = re.compile(r"(?i)(options (now available|trading) for|options begin trading|first week of .* options|interesting .* put and call options"
                  r"|noteworthy .* options|analysts predict .* gains ahead for|etf|implied .* holdings)")


class TPrices:
    """Time-MMD or FinTexTS history (side file of flow_mlp.py / fintexts.py) as 'prices': exp of the stored log level over the last value."""

    def __init__(self, nb, S=0.05):
        z = np.load(SIDE[0], allow_pickle=True)
        self.h = {(c, d): h for c, d, h in zip(z["code"], z["day"], z["hist"])}
        self.nb, self.S = nb, S

    def hist(self, code, day):
        h = self.h[(code, day)][-self.nb:]
        return np.repeat(np.exp(self.S * (h - h[-1]))[:, None], 4, 1)


class TNews:
    """Time-MMD text of a window: MM-TSFlib's Final_Search_4 at the last input step (facts dated before the forecast)."""

    def __init__(self, maxc=1500):
        z = np.load(SIDE[0], allow_pickle=True)
        self.t = {(c, d): t for c, d, t in zip(z["code"], z["day"], z["text"])}
        self.maxc = maxc

    def text(self, code, day, k=8, ns=2):
        t = str(self.t.get((code, str(day)), "")).strip()
        return t[:self.maxc] if t else "No reports or search results for this period."


SIDE = ["sel_dump/tmmd_side.npz"]   # --side: history and text of the windows (Time-MMD or FinTexTS)
TMMD_DOM = {"Algriculture": "US retail broiler composite price (monthly)", "Climate": "US precipitation (monthly)",
            "Economy": "US trade balance (monthly)", "Energy": "US gasoline price (weekly)",
            "Public_Health": "US influenza-like illness ratio (weekly)", "Security": "US FEMA grants (monthly)",
            "SocialGood": "US unemployment rate, unadjusted (monthly)", "Traffic": "US vehicle miles traveled (monthly)"}


def strip_accents(t):
    import unicodedata
    return "".join(ch for ch in unicodedata.normalize("NFKD", t) if not unicodedata.combining(ch))


class News:
    """Recent FNSPID news of one stock. --news_clean keeps items about the stock itself (its ticker in the title),
    drops option-listing and ETF boilerplate, near-duplicate titles and Zacks boilerplate sentences."""

    def __init__(self, codes, tdays, n_days, clean=0, anon=0):
        d = pd.read_parquet(f"{MEM}/news.parquet", columns=["code", "day", "title", "summary"])
        d = d[d.code.isin(set(codes))]
        self.by = {c: g for c, g in d.groupby("code")}
        # --anon hides who the stock is: its ticker and company name become "the company", so the selector cannot lean
        # on what it knows about a particular firm. Names come from NASDAQ's symbol directory and from "Name (TICKER)" titles.
        self.anon = anon
        if anon:
            nm = pd.read_csv(f"{MEM}/names.csv").set_index("code")["name"].to_dict()
            self.mask = {}
            for c, g in self.by.items():
                found = pd.Series([m.group(1) for t in g.title.values[:3000] for m in [re.search(rf"([A-Z][\w&.'\- ]{{2,40}}?)\s*\({re.escape(c)}\)", t)] if m])
                names = {nm[c]} if isinstance(nm.get(c), str) else set()
                names |= set(found.value_counts().index[:2]) if len(found) else set()
                words = sorted({n.strip() for n in names if len(n.strip()) > 2}, key=len, reverse=True)
                words = [strip_accents(w) for w in words]
                ex = rf"\((?:(?:NASDAQ|NYSE|NYSEAMERICAN|AMEX|NYSEARCA|OTC)\s*:\s*)?{re.escape(c)}\)"
                pat = "|".join([re.escape(w) + r"(?:'s)?" for w in words] + [rf"\b{re.escape(c)}\b"])
                self.mask[c] = (re.compile(ex, re.I), re.compile(pat, re.I if len(c) > 1 else 0))
        self.td, self.n, self.clean = tdays, n_days, clean

    def text(self, code, day, k=8, ns=2):
        g = self.by.get(code)
        day = np.datetime64(day)
        i = np.searchsorted(self.td, day)
        lo = self.td[max(0, i - self.n + 1)]
        if g is None:
            return "No news in the last days."
        g = g[(g.day.values >= lo) & (g.day.values <= day)].iloc[::-1]
        if self.clean:
            own = re.compile(rf"(\b|\(){re.escape(code)}(\b|\))")
            keep, seen = [], set()
            for j, t in enumerate(g.title.values):
                key = " ".join(re.findall(r"[a-z]+", t.lower())[:6])
                if own.search(t) and not SKIP.search(t) and key not in seen:
                    keep.append(j)
                    seen.add(key)
            g = g.iloc[keep]
        g = g[:k]
        if not len(g):
            return "No news about the company in the last days." if self.clean else "No news in the last days."
        out = []
        for j, (dd, t, s) in enumerate(zip(g.day.values, g.title.values, g.summary.values)):
            line = f"- [{str(dd)[5:10]}] {t}"
            if j < ns and isinstance(s, str):
                s = BOILER.sub("", s).strip() if self.clean else s
                if s:
                    line += f": {s[:300]}"
            out.append(line)
        out = "\n".join(out)
        if self.anon and code in self.mask:
            ex, pat = self.mask[code]
            out = pat.sub("the company", ex.sub("", strip_accents(out)))
            out = re.sub(r"(?i)the company(?:[\s,]+the company)+", "the company", out)
        return out


def span(hist, futs, mode):
    """Price range [lo, hi] that holds the history and all given futures, with a margin."""
    allv = np.r_[hist[:, 1:3].ravel(), *[f[:, :2].ravel() for f in futs]] if mode == "candle" else np.r_[hist[:, 3], *[f[:, 2] for f in futs]]
    lo, hi = allv.min(), allv.max()
    pad = (hi - lo) * 0.06 + 1e-6
    return lo - pad, hi + pad


def oracle_text(D, i, generic=False):
    """Diagnostic only: what actually happened over the next 10 days, in words a reader of the charts can check.
    generic (Time-MMD / FinTexTS dumps, --data tmmd): the same facts in units of the history's step size."""
    h = np.sqrt(np.arange(1, D["y"].shape[1] + 1))
    if generic:
        z = D["y"][i][:, 2].astype(np.float64) * h
        r = np.diff(np.r_[0.0, z])
        return (f"- Over the next {len(z)} steps the series will end {z[-1]:+.1f} typical steps from its last value, "
                f"with a highest point of {z.max():+.1f} and a lowest point of {z.min():+.1f}; "
                f"its typical step change will be {np.abs(r).mean():.1f}.")
    p = np.exp(D["y"][i].astype(np.float64) * float(D["sig"][i]) * h[:, None])
    c = p[:, 2]
    r = np.diff(np.r_[1.0, c])
    return (f"- In the next 10 trading days the stock will close {100 * (c[-1] - 1):+.1f}% from today, "
            f"with a high of {100 * (p[:, 0].max() - 1):+.1f}% and a low of {100 * (p[:, 1].min() - 1):+.1f}%; "
            f"its typical daily move will be {100 * np.abs(r).mean():.1f}%.")


def draw(hist, fut, mode, W=None, H=None, rng=None, band=None):
    """hist [L,4] OHLC over the last close, fut [Lh,3] (high, low, close) price over the last close.
    rng fixes the price axis (shared by all candidates of a row); by default each chart fits itself.
    band [4,Lh] = 10/25/75/90% close quantiles of all the row's candidates, drawn faintly behind this one."""
    W, H = W or globals()["W"], H or globals()["H"]
    L, Lh = len(hist), len(fut)
    op = np.r_[hist[-1, 3], fut[:-1, 2]]
    fut4 = np.c_[op, fut[:, 0], fut[:, 1], fut[:, 2]]
    lo, hi = rng or span(hist, [fut], mode)
    n = L + Lh
    bw = W / (n + 1)
    X = lambda i: (i + 1) * bw
    Y = lambda p: (hi - p) / (hi - lo) * (H - 1)
    im = Image.new("RGB", (W, H), BG)
    g = ImageDraw.Draw(im)
    g.rectangle([X(L) - bw / 2, 0, W, H], fill=SHADE)
    if band is not None:
        xs = [X(L - 1)] + [X(L + i) for i in range(Lh)]
        c0 = hist[-1, 3]
        for (a, b), col in (((0, 3), (205, 214, 228)), ((1, 2), (178, 192, 214))):
            top = [(x, Y(v)) for x, v in zip(xs, np.r_[c0, band[b]])]
            bot = [(x, Y(v)) for x, v in zip(xs, np.r_[c0, band[a]])][::-1]
            g.polygon(top + bot, fill=col)
    if mode == "candle":
        for i, (o, h, l, c) in enumerate(np.r_[hist, fut4]):
            col = (UP if c >= o else DN)
            x = X(i)
            g.line([x, Y(h), x, Y(l)], fill=col, width=1)
            y0, y1 = sorted((Y(o), Y(c)))
            g.rectangle([x - bw * 0.35, y0, x + bw * 0.35, max(y1, y0 + 1)], fill=col)
    else:
        pts = [(X(i), Y(c)) for i, c in enumerate(hist[:, 3])]
        g.line(pts, fill=INK, width=2)
        g.line([pts[-1]] + [(X(L + i), Y(c)) for i, c in enumerate(fut[:, 2])], fill=FUT, width=2)
    return im


class Scorer(nn.Module):
    def __init__(self, a, dev):
        super().__init__()
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(a.vlm, dtype=torch.bfloat16, attn_implementation="sdpa")
        m.requires_grad_(False)
        # The decoder without its vocabulary head: hidden states at each image's end and the last token are scored.
        self.core = m.model
        if a.lora:
            from peft import LoraConfig, get_peft_model
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            m.enable_input_require_grads()
            cfg = LoraConfig(r=a.lora, lora_alpha=2 * a.lora, lora_dropout=0.05,
                             target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
                             + (r"|.*visual.*\.(qkv|attn\.proj|gate_proj|up_proj|down_proj|linear_fc1|linear_fc2|mlp\.0|mlp\.2)" if a.lora_vision else ""))
            m = get_peft_model(m, cfg)
        # from_pretrained returns eval mode, which would switch gradient checkpointing off.
        self.vlm = m.to(dev).train()
        d = m.config.text_config.hidden_size if hasattr(m.config, "text_config") else m.config.hidden_size
        # One marker token after each candidate's label in the prompt; its hidden state is the candidate state.
        from transformers import AutoTokenizer
        self.vend = AutoTokenizer.from_pretrained(a.vlm).convert_tokens_to_ids(MARK)
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512)).to(dev)
        self.kind = nn.Parameter(torch.zeros(2, 512, device=dev))
        lay = nn.TransformerEncoderLayer(512, 8, 1024, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.mix = nn.TransformerEncoder(lay, 2).to(dev)
        self.out = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 1)).to(dev)
        # Optional numeric summary of each drawn candidate (no text): its centrality among the draws, volatility and return.
        # --prior 2 also gives the candidate's whole (high, low, close) path as numbers.
        self.npri = (4, NPRI)[a.prior - 1] if a.prior else 0
        self.pri = nn.Sequential(nn.Linear(self.npri, 512), nn.GELU(), nn.Linear(512, 512)).to(dev) if a.prior else None
        # Auxiliary chart reading: each candidate's state predicts that panel's own centrality, volatility and return.
        self.aux = nn.Linear(d, 3).to(dev) if a.aux else None
        # Auxiliary news reading: the last token predicts the truth's 10-day return, its size, vol and range (no candidate).
        self.auxn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4)).to(dev) if a.aux_news else None
        self.a, self.dev = a, dev

    def head_params(self):
        return [*self.proj.parameters(), self.kind, *self.mix.parameters(), *self.out.parameters(),
                *(self.aux.parameters() if self.aux is not None else []),
                *(self.auxn.parameters() if self.auxn is not None else []),
                *(self.pri.parameters() if self.pri is not None else [])]

    def forward(self, b):
        b = {k: v.to(self.dev, non_blocking=True) for k, v in b.items()}
        pr = b.pop("prior")
        yt = b.pop("ytrue", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.core(**b, use_cache=False).last_hidden_state
        B = h.shape[0]
        c = h[b["input_ids"] == self.vend].view(B, -1, h.shape[-1])
        self.aux_loss = F.mse_loss(self.aux(c.float()), pr[..., 1:4]) if self.aux is not None else None
        self.auxn_loss = F.smooth_l1_loss(self.auxn(h[:, -1].float()), yt) if self.auxn is not None and yt is not None else None
        zc = self.proj(c.float()) + self.kind[1]
        if self.pri is not None:
            zc = zc + self.pri(pr[..., :self.npri])
        z = torch.cat([self.proj(h[:, -1:].float()) + self.kind[0], zc], 1)
        return self.out(self.mix(z)[:, 1:])[..., 0]


NPRI = 34


def prior(D, i, ks):
    """[len(ks), NPRI]: mean close distance to all of the row's draws (raw and z-scored in the row), daily vol, 10-day return,
    then the path itself."""
    c = D["draws"][i, :, :, 2].astype(np.float32)
    cen = np.abs(c[:, None] - c[None]).mean((1, 2))
    f = features(D["draws"][i].astype(np.float32))
    z = (cen - cen.mean()) / (cen.std() + 1e-6)
    return np.concatenate([np.stack([cen, z, f[:, 0], f[:, 4] / 3], 1), D["draws"][i].reshape(len(c), -1)], 1)[ks].astype(np.float32)


def candidates(D, i, ks, prices, a):
    """One grid image of numbered panels, one panel per candidate, on a shared value axis (--yshare 1)."""
    sig = float(D["sig"][i])
    h = np.sqrt(np.arange(1, D["draws"].shape[2] + 1))[:, None]
    hist = prices.hist(D["code"][i], D["day"][i])
    futs = [np.exp(D["draws"][i, k].astype(np.float64) * sig * h) for k in ks]
    rng = span(hist, futs, a.mode) if a.yshare else None
    band = None
    if a.band:
        allc = np.exp(D["draws"][i, :, :, 2].astype(np.float64) * sig * h[:, 0])
        band = np.quantile(allc, [0.1, 0.25, 0.75, 0.9], axis=0)
    cols = GRID_COLS
    rows = -(-len(futs) // cols)
    im = Image.new("RGB", (cols * W, rows * H), BG)
    g = ImageDraw.Draw(im)
    for j, f in enumerate(futs):
        x, y = (j % cols) * W, (j // cols) * H
        im.paste(draw(hist, f, a.mode, rng=rng, band=band), (x, y))
        g.rectangle([x, y, x + W - 1, y + H - 1], outline=(150, 150, 150))
        g.rectangle([x + 1, y + 1, x + 19, y + 13], fill=(255, 255, 255))
        g.text((x + 3, y + 2), str(j + 1), fill=(0, 0, 0))
    return [im]


GRID_COLS = 4
MARK = "<|box_end|>"


def num_text(D, i, ks, prices, a):
    """Control for "why a chart": the same history and candidates written as numbers, no image.
    --input num puts every close on the chart's own shared price axis as an integer 0-99 (the normalization the
    chart applies by construction); --input numpct writes closes as % from the last close (one decimal)."""
    sig = float(D["sig"][i])
    h = np.sqrt(np.arange(1, D["draws"].shape[2] + 1))[:, None]
    hist = prices.hist(D["code"][i], D["day"][i])
    futs = [np.exp(D["draws"][i, k].astype(np.float64) * sig * h) for k in ks]
    if a.input == "num":
        lo, hi = span(hist, futs, "line")
        f = lambda v: " ".join(str(int(round(x))) for x in np.clip((v - lo) / (hi - lo) * 99, 0, 99))
        unit = "Values are on one shared scale from 0 (lowest) to 99 (highest)."
    else:
        f = lambda v: " ".join(f"{100 * (x - 1):+.1f}" for x in v)
        unit = "Values are closes in % from the last close."
    out = f"{unit}\nHistory (last {len(hist)} closes): {f(hist[:, 3])}"
    return out, [f(fu[:, 2]) for fu in futs]


FLIP = {"rise": "fall", "rises": "falls", "rising": "falling", "rose": "fell", "risen": "fallen", "increase": "decrease",
        "increases": "decreases", "increased": "decreased", "increasing": "decreasing", "up": "down", "higher": "lower",
        "highest": "lowest", "high": "low", "gain": "loss", "gains": "losses", "gained": "lost", "surge": "plunge",
        "surges": "plunges", "surged": "plunged", "jump": "drop", "jumps": "drops", "jumped": "dropped", "climb": "decline",
        "climbs": "declines", "climbed": "declined", "grow": "shrink", "grows": "shrinks", "grew": "shrank", "growth": "contraction",
        "strong": "weak", "stronger": "weaker", "strength": "weakness", "positive": "negative", "bullish": "bearish",
        "beat": "miss", "beats": "misses", "upgrade": "downgrade", "upgrades": "downgrades", "upgraded": "downgraded",
        "buy": "sell", "outperform": "underperform", "record": "weak", "boost": "cut", "boosts": "cuts", "boosted": "cut",
        "above": "below", "more": "less", "improve": "worsen", "improved": "worsened", "improves": "worsens", "rally": "selloff",
        "expand": "contract", "expanded": "contracted", "expansion": "contraction", "wetter": "drier", "warmer": "cooler"}
FLIP.update({v: k for k, v in list(FLIP.items()) if v not in FLIP})
_FLIP_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, FLIP), key=len, reverse=True)) + r")\b", re.I)


def flip_text(t):
    """Contradicting control: every direction word of the text replaced by its opposite (rise <-> fall, beat <->
    miss, ...), so the text keeps its topic and form but points the other way."""
    def f(m):
        w = m.group(0); o = FLIP[w.lower()]
        return o.upper() if w.isupper() else (o.capitalize() if w[0].isupper() else o)
    return _FLIP_RE.sub(f, t)


class Feeder:
    """Builds processed VLM batches; runs inside DataLoader workers so rendering overlaps the GPU."""

    def __init__(self, prices, news, a):
        from transformers import AutoProcessor
        px = 28 * 28
        self.proc = AutoProcessor.from_pretrained(a.vlm, size={"shortest_edge": px, "longest_edge": W * H * 64}, backend="torchvision")
        self.proc.tokenizer.padding_side = "left"
        self.prices, self.news, self.a = prices, news, a

    def ctrl_text(self, D, i, src):
        """Text of row src, or under --ctrl (evaluation-only controls): "none" an empty report, "foreign" a text of
        an unrelated domain (a Time-MMD report of another series), chosen per row with a fixed seed."""
        c = getattr(self, "ctrl", "")
        if c == "none":
            return "No reports or search results for this period." if self.a.data == "tmmd" else "No news in the last days."
        if c == "flip":
            return flip_text(self.news.text(D["code"][src], D["day"][src], self.a.max_news, self.a.n_summary))
        if c == "foreign":
            if not hasattr(self, "pool"):
                z = np.load("sel_dump/tmmd_side.npz", allow_pickle=True)
                self.pool = [(str(cc), str(t)[:1500]) for cc, t in zip(z["code"], z["text"]) if str(t).strip()]
            own = str(D["code"][i])
            r = np.random.default_rng([int(i), 7])
            while True:
                cc, t = self.pool[int(r.integers(len(self.pool)))]
                if cc != own:
                    return t
        return self.news.text(D["code"][src], D["day"][src], self.a.max_news, self.a.n_summary)

    def prompt(self, D, i, K, src=None, kk=None):
        src = i if src is None else src
        if self.a.input not in ("chart", "chartnum"):
            return self.prompt_num(D, i, kk, src)
        code = D["code"][i]
        if self.a.text == "oracle":
            nw = oracle_text(D, src, self.a.data == "tmmd")
        else:
            nw = None if self.a.text == "none" else self.ctrl_text(D, i, src)
        if self.a.data == "tmmd":
            hz = D["draws"].shape[2]
            name = f"Time series: {TMMD_DOM.get(code, code)}" if code in TMMD_DOM else self.a.series_desc.format(code)
            head = f"{name}. Related reports and facts available before the forecast:\n{nw}\n\n" if nw is not None else f"{name}.\n\n"
            head += (f"The image has {K} numbered panels on the same scale. Each shows the same last {self.a.hist} observations "
                     f"and a different possible path for the next {hz} steps (shaded).")
            content = [{"type": "text", "text": head}, {"type": "image"},
                       {"type": "text", "text": "".join(f"\nCandidate {j + 1}{MARK}" for j in range(K))},
                       {"type": "text", "text": "\nWhich candidate is the most plausible future given the chart and the text?"}]
            return self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        name = "A stock" if self.a.anon else f"Stock {code}"
        head = f"{name}. News of the last days, newest first:\n{nw}\n\n" if nw is not None else f"{name}.\n\n"
        head += (f"The image has {K} numbered panels on the same price scale. Each shows the same last {self.a.hist} trading days "
                 "and a different possible path for the next 10 days (shaded).")
        lab = [""] * K
        if self.a.input == "chartnum" and kk is not None:
            # Chart plus the same candidates as % numbers: the image gives the shape, the numbers the scale.
            hs, cs = num_text(D, i, kk, self.prices, self.a)
            head += "\n" + hs
            lab = [f": {c}" for c in cs]
        content = [{"type": "text", "text": head}, {"type": "image"},
                   {"type": "text", "text": "".join(f"\nCandidate {j + 1}{lab[j]}{MARK}" for j in range(K))}]
        content.append({"type": "text", "text": "\nWhich candidate is the most plausible future given the chart and the news?"})
        return self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)

    def prompt_num(self, D, i, kk, src):
        """Same question as the grid prompt, with the series and the candidates as number text (no image)."""
        code = D["code"][i]
        if self.a.text == "oracle":
            nw = oracle_text(D, src, self.a.data == "tmmd")
        else:
            nw = None if self.a.text == "none" else self.ctrl_text(D, i, src)
        hz = D["draws"].shape[2]
        if self.a.data == "tmmd":
            name, what, ev = (f"Time series: {TMMD_DOM.get(code, code)}" if code in TMMD_DOM else self.a.series_desc.format(code)), "Related reports and facts available before the forecast", "text"
        else:
            name, what, ev = ("A stock" if self.a.anon else f"Stock {code}"), "News of the last days, newest first", "news"
        head = f"{name}. {what}:\n{nw}\n\n" if nw is not None else f"{name}.\n\n"
        hs, cs = num_text(D, i, kk, self.prices, self.a)
        head += f"{hs}\nBelow are {len(kk)} candidate paths for the next {hz} steps, written the same way."
        body = "".join(f"\nCandidate {j + 1}: {c}{MARK}" for j, c in enumerate(cs))
        content = [{"type": "text", "text": head + body + f"\nWhich candidate is the most plausible future given the series and the {ev}?"}]
        return self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)

    def build(self, D, rows, ks, src=None):
        """rows with a candidate order each -> processor tensors, K images per row."""
        texts, ims = [], []
        for i, kk in zip(rows, ks):
            texts.append(self.prompt(D, i, len(kk), None if src is None else src[i], kk))
            if self.a.input in ("chart", "chartnum"):
                ims += candidates(D, i, kk, self.prices, self.a)
        out = dict(self.proc(text=texts, images=ims or None, return_tensors="pt", padding=True))
        out["prior"] = torch.tensor(np.stack([prior(D, i, kk) for i, kk in zip(rows, ks)]), dtype=torch.float32)
        return out


class TrainStream(torch.utils.data.IterableDataset):
    def __init__(self, feeder, D, seed):
        self.f, self.D, self.seed = feeder, D, seed

    def __iter__(self):
        w = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, w.id if w else 0])
        a, D = self.f.a, self.D
        while True:
            idx = rng.choice(len(D["day"]), a.bs, p=None)
            # A random subset of k candidates in a random panel order.
            ks = [rng.choice(D["draws"].shape[1], a.k, replace=False) for i in idx]
            tg = [target_dist(D["draws"][i, k].astype(np.float32), D["y"][i].astype(np.float32), a) for i, k in zip(idx, ks)]
            b = self.f.build(D, idx, ks)
            if a.aux_news:
                y = D["y"][idx].astype(np.float32)
                f = features(y)
                b["ytrue"] = torch.tensor(np.stack([y[:, -1, 2], np.abs(y[:, -1, 2]), f[:, 0], f[:, 6]], 1))
            yield b, torch.tensor(np.stack(tg))


class EvalStream(torch.utils.data.IterableDataset):
    def __init__(self, feeder, D, rows, K, src, bs, seed):
        # Each row shows its first K candidates in a fixed random order.
        rng = np.random.default_rng(seed)
        self.order = [rng.permutation(K) for i in rows]
        js = np.arange(len(rows))
        self.chunks = [js[s:s + bs] for s in range(0, len(rows), bs)]
        self.f, self.D, self.rows, self.src = feeder, D, rows, src

    def __iter__(self):
        w = torch.utils.data.get_worker_info()
        for c in self.chunks[(w.id if w else 0)::(w.num_workers if w else 1)]:
            yield self.f.build(self.D, self.rows[c], [self.order[j] for j in c], self.src), torch.tensor(c), torch.tensor(np.stack([self.order[j] for j in c]))


def scores(model, feeder, D, rows, K, src=None, bs=4, workers=6, seed=0):
    """Scores [len(rows), K] for the first K candidates of each row; src maps a row to the row whose news it reads."""
    out = np.zeros((len(rows), K), np.float32)
    model.eval()
    dl = torch.utils.data.DataLoader(EvalStream(feeder, D, np.asarray(rows), K, src, bs, seed), batch_size=None, num_workers=workers)
    with torch.no_grad():
        for b, js, order in dl:
            out[js.numpy()[:, None], order.numpy()] = model(b).float().cpu().numpy()
    model.train()
    return out


SHAPE = [0, 2, 3, 6]  # vol, max drawdown, max run-up, range among sel_train.features
SHAPE_SD = np.array([0.4, 2.2, 1.65, 0.4], np.float32)  # rough spread of the truth's values, so each term weighs alike


def target_dist(dr, y, a):
    """[K] shape-aware distance d_k of each candidate path [K,Lh,3] (high, low, close) to the truth [Lh,3]:
    MAE of the close path + w_shape * mean scaled error of four shape properties (volatility, largest drop, largest rise,
    high-low range; the range term is zero where the data give no range) + w_ret * endpoint error (paper: lambda_end)."""
    e = np.abs(dr - y[None]).mean(1)
    d = e[:, 2]
    if a.w_shape:
        f, ft = features(dr)[:, SHAPE], features(y[None])[:, SHAPE]
        d = d + a.w_shape * (np.abs(f - ft) / SHAPE_SD).mean(1)
    d = d + a.w_ret * np.abs(dr[:, -1, 2] - y[-1, 2])
    return d


def main():
    global W, H
    q = argparse.ArgumentParser()
    q.add_argument("--train", required=True)
    q.add_argument("--val", required=True)
    q.add_argument("--test", default="", help="name=glob;name2=glob2")
    q.add_argument("--mode", default="candle", choices=("candle", "line"))
    q.add_argument("--text", default="news", choices=("news", "none", "oracle"),
                   help="oracle: a diagnostic text that describes the true next 10 days, to test whether the VLM can ground text in the charts")
    q.add_argument("--news_clean", type=int, default=0)
    q.add_argument("--anon", type=int, default=0, help="Hide the ticker and company name from the prompt and the news")
    q.add_argument("--news_days", type=int, default=5)
    q.add_argument("--max_news", type=int, default=8)
    q.add_argument("--n_summary", type=int, default=2, help="How many of the newest items also show their summary")
    q.add_argument("--lora", type=int, default=16)
    q.add_argument("--tau", type=float, default=0.1)
    q.add_argument("--sel_w_shape", type=float, default=-1.0, help="Shape weight of the validation target used for model selection (default: --w_shape), so runs trained with different weights compare on one target")
    q.add_argument("--w_shape", type=float, default=0.0, help="Weight of the path-shape mismatch (vol, drawdown, run-up, range) in the target")
    q.add_argument("--vlm", default=VLM)
    q.add_argument("--img", default=f"{W}x{H}", help="Size WxH of one panel of the grid image")
    q.add_argument("--lora_vision", type=int, default=0, help="Also put LoRA on the vision tower, which never saw charts like ours")
    q.add_argument("--aux", type=float, default=0.0, help="Weight of the auxiliary chart-reading loss")
    q.add_argument("--aux_news", type=float, default=0.0, help="Weight of the last-token head predicting the truth's 10-day return, size, vol, range")
    q.add_argument("--hist", type=int, default=NB, help="History bars before the 10 future bars")
    q.add_argument("--band", type=int, default=0, help="Draw the spread of all the row's candidates behind each one")
    q.add_argument("--yshare", type=int, default=0, help="Draw a row's candidates on one shared price axis")
    q.add_argument("--obj", default="ce", choices=("ce", "reg"))
    q.add_argument("--prior", type=int, default=0)
    q.add_argument("--input", default="chart", choices=("chart", "num", "numpct", "chartnum"),
                   help="Control: write the history and candidates as numbers in the prompt instead of the grid image")
    q.add_argument("--k", type=int, default=8, help="Candidates per row in a training step")
    q.add_argument("--k_eval", type=int, default=16)
    q.add_argument("--bs", type=int, default=4, help="Rows per step")
    q.add_argument("--lr", type=float, default=1e-4)
    q.add_argument("--lr_head", type=float, default=1e-3)
    q.add_argument("--steps", type=int, default=6000)
    q.add_argument("--ctrl", default="", help="Evaluation-only text controls, comma-separated: none (empty report), foreign (unrelated-domain text), flip (direction words reversed)")
    q.add_argument("--ctrl_only", type=int, default=0, help="With --ctrl, skip the shuffled-text control")
    q.add_argument("--eval_every", type=int, default=1000)
    q.add_argument("--val_rows", type=int, default=1500)
    q.add_argument("--test_rows", type=int, default=20000)
    q.add_argument("--workers", type=int, default=6)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--w_ret", type=float, default=0.0, help="Weight of the endpoint error in the target distance (lambda_end)")
    q.add_argument("--no_val", type=int, default=0, help="Skip the final val eval (test-only runs of a saved model)")
    q.add_argument("--load", default="", help="Start from a saved run (selector_runs/*.pt): the FNSPID selector initializes the FinTexTS and Time-MMD ones; with --steps 0 this only evaluates it")
    q.add_argument("--tag", required=True)
    q.add_argument("--side", default="sel_dump/tmmd_side.npz", help="--data tmmd: side file with each window's history and text")
    q.add_argument("--series_desc", default="Time series: {}", help="--data tmmd: name of a series not in TMMD_DOM ({} = code)")
    q.add_argument("--data", default="fnspid", choices=("fnspid", "tmmd"), help="tmmd: Time-MMD or FinTexTS dumps (flow_mlp.py, fintexts.py)")
    a = q.parse_args()
    W, H = map(int, a.img.split("x"))
    dev = "cuda"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    TR, VA = load(a.train), load(a.val)
    TE = {n: load(p) for n, p in (x.split("=") for x in a.test.split(";") if x)}
    codes = set(TR["code"]) | set(VA["code"]) | {c for D in TE.values() for c in D["code"]}
    SIDE[0] = a.side
    if a.data == "tmmd":
        prices, news = TPrices(a.hist), TNews()
    else:
        prices = Prices(codes, a.hist)
        td = pd.to_datetime(pd.read_csv(f"{MEM}/1d/trade_dates.csv").date.astype(str)).values.astype("datetime64[D]")
        news = News(codes, td, a.news_days, a.news_clean, a.anon)
    feeder = Feeder(prices, news, a)
    model = Scorer(a, dev)
    params = [p for p in model.vlm.parameters() if p.requires_grad]
    print("trainable vlm params", sum(p.numel() for p in params), flush=True)
    opt = torch.optim.AdamW([{"params": params, "lr": a.lr}, {"params": model.head_params(), "lr": a.lr_head}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * max(0.05, 1 - s / max(a.steps, 1)))
    # The same validation rows for every seed, so runs compare on equal footing.
    vrows = np.random.default_rng(0).choice(len(VA["day"]), min(a.val_rows, len(VA["day"])), replace=False)
    best, best_state, t0 = 1e9, None, time.time()
    if a.load:
        best_state = torch.load(a.load)["state"]
        model.load_state_dict(best_state, strict=False)
    train = iter(torch.utils.data.DataLoader(TrainStream(feeder, TR, a.seed), batch_size=None, num_workers=a.workers,
                                             prefetch_factor=4, persistent_workers=True)) if a.steps else None
    for step in range(1, a.steps + 1):
        b, tg = next(train)
        tg = tg.to(dev)
        s = model(b).view(a.bs, a.k)
        loss = objective(s.float(), tg, a)
        if model.aux_loss is not None:
            loss = loss + a.aux * model.aux_loss
        if model.auxn_loss is not None:
            loss = loss + a.aux_news * model.auxn_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f} {(time.time() - t0) / step:.2f}s/step", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            sc = scores(model, feeder, VA, vrows, a.k_eval, workers=a.workers)
            p = sc.argmax(1)
            c = VA["draws"][vrows, :a.k_eval, :, 2].astype(np.float32)
            err = np.abs(c - VA["y"][vrows, None, :, 2]).mean(-1)
            mae = err[np.arange(len(p)), p].mean()
            # Model selection uses the training target itself (MAE alone unless --w_shape / --w_ret are set).
            asel = argparse.Namespace(**{**vars(a), "w_shape": a.w_shape if a.sel_w_shape < 0 else a.sel_w_shape})
            td = np.stack([target_dist(VA["draws"][i, :a.k_eval].astype(np.float32), VA["y"][i].astype(np.float32), asel) for i in vrows])
            v = td[np.arange(len(p)), p].mean()
            med = np.abs(c[:, :, None] - c[:, None]).mean(-1).sum(-1).argmin(1)
            print(f"step {step} val mae sel {mae:.4f} random {err.mean():.4f} oracle {err.min(1).mean():.4f} target sel {v:.4f} random {td.mean():.4f} medoid {err[np.arange(len(p)), med].mean():.4f}", flush=True)
            if v < best:
                best = v
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items() if "lora" in k or k.split(".")[0] in ("proj", "kind", "mix", "out", "pri", "scale", "aux", "auxn")}
    model.load_state_dict(best_state, strict=False)
    os.makedirs("selector_runs", exist_ok=True)
    if a.steps:
        torch.save(dict(state=best_state, args=vars(a)), f"selector_runs/{a.tag}.pt")
    out = dict(args=vars(a), val_best=float(best))
    sets = ([] if a.no_val else [("val", VA)]) + list(TE.items())
    for name, D, n in [(k, D, a.test_rows) for k, D in sets]:
        # Fixed eval rows for every seed, so runs compare on the same rows.
        rows = np.sort(np.random.default_rng(0).choice(len(D["day"]), min(n, len(D["day"])), replace=False))
        S = {k: v[rows] for k, v in D.items()}
        S["draws"] = S["draws"][:, :a.k_eval]
        sc = scores(model, feeder, S, np.arange(len(rows)), a.k_eval, workers=a.workers)
        print(f"\n== {name} ==")
        res = report(S, evaluate(S, sc.argmax(1)))
        np.savez(f"selector_runs/{a.tag}_{name}_scores.npz", rows=rows, scores=sc)
        for c in [x for x in a.ctrl.split(",") if x]:
            feeder.ctrl = c
            sc_c = scores(model, feeder, S, np.arange(len(rows)), a.k_eval, workers=a.workers)
            feeder.ctrl = ""
            print(f"\n== {name}, control text: {c} ==")
            res[f"ctrl_{c}"] = report(S, evaluate(S, sc_c.argmax(1)))
            np.savez(f"selector_runs/{a.tag}_{name}_ctrl_{c}_scores.npz", rows=rows, scores=sc_c)
        if a.text == "news" and not a.ctrl_only:
            perm = rng.permutation(len(rows))
            if a.data == "tmmd":
                # Text from another window of the same series, so the control keeps the domain and differs only in time.
                perm = np.arange(len(rows))
                for c in np.unique(S["code"]):
                    m = np.flatnonzero(S["code"] == c)
                    perm[m] = rng.permutation(m)
            sh = scores(model, feeder, S, np.arange(len(rows)), a.k_eval, src=perm, workers=a.workers)
            print(f"\n== {name}, news from another row ==")
            res["shuffle"] = report(S, evaluate(S, sh.argmax(1)))
            np.savez(f"selector_runs/{a.tag}_{name}_shuffle_scores.npz", rows=rows, scores=sh, perm=perm)
        out[name] = res
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(f"results/{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
