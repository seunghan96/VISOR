"""Data and model code of the Chart-Native predictor of VISOR on FNSPID (KiT-architecture flow over a candle canvas).

The model follows the KiT diffusion transformer (https://github.com/Luciferbobo/KiT, imported from KIT_ROOT, never
copied here): the input is a candle chart image of the last 90 daily bars (--chart), and the output is a canvas of
--pix value bins per future day (one half marks the close bin, the other the bins between the low and the high).
Windows follow KiT's definitions in `kit` mode. The FNSPID candles are read from the flat arrays written by
scripts/data/build_kit_mem.py (VISOR_MEM). Other modes and markets are research options not used in the paper.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from visor import paths

KIT = paths.KIT
sys.path.insert(0, KIT)
from kit_v1.config import Config  # noqa: E402
from kit_v1.data.normalize import load_stats, stats_key  # noqa: E402
from kit_v1.data.calendar import (MARKET_IDS, TIMESCALE_IDS, TIMESCALE_MINUTES, bars_per_day,  # noqa: E402
                                  gen_calendar, gen_future_timestamps, load_trading_days)
from kit_v1.model import KlineDiT  # noqa: E402

MEM = paths.MEM
ROOTS = {"ashare": f"{KIT}/data", "us": f"{paths.DATA}/kit_us",
         "crypto": f"{paths.DATA}/kit_crypto", "cnq": f"{MEM}/cnq/1d", "fnspid": f"{MEM}/fnspid/1d",
         "yahoo": f"{MEM}/yahoo/1d", "yahoo_jp": f"{MEM}/yahoo_jp/1d", "yahoo_hk": f"{MEM}/yahoo_hk/1d"}
# Qlib daily data and FNSPID are separate trees of the same markets; Yahoo price-only trees (build_yahoo.py) reuse the
# US market id, since they only pretrain the text-free generator.
ALIAS = {"cnq": "ashare", "fnspid": "us", "yahoo": "us", "yahoo_jp": "us", "yahoo_hk": "us"}
SCALES = ("1m", "5m", "15m", "30m", "1h", "2h", "1d")
WINDOWS = {"1m": (1200, 120, 300), "5m": (960, 96, 240), "15m": (800, 64, 200), "30m": (640, 40, 160),
           "1h": (400, 20, 100), "2h": (360, 20, 90), "1d": (250, 20, 64)}
PYRAMID = {"1d": 120, "2h": 60, "1h": 60, "30m": 78, "15m": 100, "5m": 160, "1m": 240}
DAY0 = np.datetime64("1970-01-01", "D")
FEATURES = ("r_gap", "r_body", "r_up", "r_dn", "v")


def to_minutes(ts):
    return int(pd.Timestamp(ts).value // 60_000_000_000)


class Calendar:
    """Table lookups that reproduce gen_calendar without touching pandas per window."""

    def __init__(self, market, scale):
        self.market, self.scale = market, scale
        self.shift = 1 if market == "crypto" else 0
        days = pd.date_range("2000-01-01", "2030-12-31", freq="D")
        cal = gen_calendar(days + pd.Timedelta(hours=12), "us", "1d")
        self.day0 = (days[0].to_datetime64().astype("datetime64[D]") - DAY0).astype(int)
        self.dow, self.month, self.yday = cal["dow"], cal["month"], cal["yday"]
        grid = pd.date_range("2024-01-02", periods=1441, freq="min")
        grid = grid if market == "crypto" else grid[:1440]
        if scale == "1d":
            c = gen_calendar(grid[:1], market, scale)
            self.clock = np.repeat(c["clock"], 1441, 0)
            self.sess = np.repeat(c["sess"], 1441, 0)
        else:
            mins = TIMESCALE_MINUTES[scale]
            from kit_v1.data.calendar import _daily_grid_minutes
            on = np.array(_daily_grid_minutes(market, scale))
            c = gen_calendar(pd.Timestamp("2024-01-02") + pd.to_timedelta(on, "min"), market, scale)
            self.clock = np.zeros((1441, 16), np.float32)
            self.sess = np.zeros((1441, 2), np.float32)
            self.clock[on % 1440] = c["clock"]
            self.clock[on] = c["clock"]
            self.sess[on % 1440] = c["sess"]
            self.sess[on] = c["sess"]

    def __call__(self, minutes):
        """minutes [L] int64 close times -> dict of per-bar calendar arrays."""
        m = np.asarray(minutes, np.int64)
        day = (m - self.shift) // 1440 - self.day0
        mod = m % 1440
        if self.market == "crypto":
            mod = np.where(mod == 0, 1440, mod)
        return {"clock": self.clock[mod], "sess": self.sess[mod], "dow": self.dow[day],
                "month": self.month[day], "yday": self.yday[day]}


class Series:
    """Memory-mapped arrays for one market and scale."""

    def __init__(self, market, scale):
        d = f"{MEM}/{market}/{scale}"
        self.X = np.load(f"{d}/X.npy", mmap_mode="r")
        self.M = np.load(f"{d}/M.npy", mmap_mode="r")
        self.E = np.load(f"{d}/E.npy", mmap_mode="r")
        self.O = np.load(f"{d}/O.npy", mmap_mode="r")
        idx = pd.read_parquet(f"{d}/index.parquet")
        self.codes = idx.code.tolist()
        self.start, self.end = idx.start.values, idx.end.values
        self.vocab, self.sector = idx.vocab.values, idx.sector.values
        self.pos = {c: i for i, c in enumerate(self.codes)}
        self.cal = Calendar(ALIAS.get(market, market), scale)
        self.minutes = TIMESCALE_MINUTES[scale]

    def upto(self, i, t):
        """First array position after instrument i's last bar with close <= t."""
        s, e = self.start[i], self.end[i]
        return s + int(np.searchsorted(self.M[s:e], t, side="right"))


class Mem:
    def __init__(self, markets, scales=SCALES):
        # The Qlib tree has no intraday scales, so a pooled run skips the pairs that do not exist.
        self.s = {(m, sc): Series(m, sc) for m in markets for sc in scales if os.path.isdir(f"{MEM}/{m}/{sc}")}
        self.tdays = {m: load_trading_days(f"{ROOTS[m]}/trade_dates.csv") for m in markets}
        stats = load_stats(f"{KIT}/ckpt/hf/stats.json")
        self.unit = {(m, sc): np.array([stats[stats_key(ALIAS.get(m, m), sc, n)] for n in FEATURES], np.float32) for m, sc in self.s}

    def pyramid(self, market, code, t, mult=1.0, scales=SCALES, days=0):
        """Candles from coarse to fine ending at minute t, no two sharing any time span.

        days overrides the daily count, so a 1d target can keep KiT's 250-day reach.
        """
        blocks, end = [], t
        for sc in scales:
            n = days if sc == "1d" and days else int(round(PYRAMID[sc] * mult))
            if n == 0:
                continue
            ser = self.s[market, sc]
            i = ser.pos.get(code)
            if i is None:
                continue
            hi = ser.upto(i, end)
            lo = max(ser.start[i], hi - n)
            if hi <= lo:
                continue
            blocks.append((sc, lo, hi))
            end = int(ser.M[lo]) - ser.minutes
        return blocks[::-1]

    def window(self, market, scale, i, end, Lc, Lh, mode="kit", mult=1.0, fut=True):
        """History ending at position end (exclusive) and the Lh bars after it.

        With fut=False the target calendar comes from the exchange calendar and no future
        data is read, which is what inference must do.
        """
        ser = self.s[market, scale]
        t = int(ser.M[end - 1])
        if mode.startswith("pyr"):
            days = int(mode[3:] or 0)
            blocks = self.pyramid(market, ser.codes[i], t, mult, days=days)
            n = int(sum(days if s == "1d" and days else round(PYRAMID[s] * mult) for s in SCALES))
            xs, ms, es, sc_ids, cals = [], [], [], [], []
            for sc, lo, hi in blocks:
                b = self.s[market, sc]
                xs.append(np.asarray(b.X[lo:hi]))
                ms.append(np.asarray(b.M[lo:hi]))
                es.append(np.asarray(b.E[lo:hi]))
                sc_ids.append(np.full(hi - lo, TIMESCALE_IDS[sc]))
                cals.append(b.cal(ms[-1]))
            x = np.concatenate(xs)
            ev = np.concatenate(es).astype(np.int64)
            sid = np.concatenate(sc_ids)
            cal = {k: np.concatenate([c[k] for c in cals]) for k in cals[0]}
            Lc = n
        else:
            lo = max(ser.start[i], end - Lc)
            x, m, ev = np.asarray(ser.X[lo:end]), np.asarray(ser.M[lo:end]), np.asarray(ser.E[lo:end]).astype(np.int64)
            sid = np.full(end - lo, TIMESCALE_IDS[scale])
            cal = ser.cal(m)
        if fut:
            tgt = np.asarray(ser.X[end:end + Lh])
            fm = np.asarray(ser.M[end:end + Lh]).astype(np.int64)
        else:
            tgt = np.zeros((Lh, 5), np.float32)
            stamp = pd.Timestamp(t * 60_000_000_000)
            fm = gen_future_timestamps(stamp, Lh, ALIAS.get(market, market), scale, trading_days=self.tdays[market]).asi8 // 60_000_000_000
        fc = ser.cal(fm)
        c = len(x)
        pad = Lc - c
        L = Lc + Lh
        out = {"x": np.zeros((L, 5), np.float32), "event": np.zeros(L, np.int64), "pad_mask": np.ones(L, bool),
               "tok_scale": np.full(L, TIMESCALE_IDS[scale], np.int64)}
        out["x"][pad:Lc], out["x"][Lc:] = x, tgt
        out["event"][pad:Lc] = ev
        out["tok_scale"][pad:Lc] = sid
        out["pad_mask"][:pad] = False
        for k, shape in (("clock", (16,)), ("sess", (2,)), ("yday", (8,)), ("dow", ()), ("month", ())):
            a = np.zeros((L,) + shape, cal[k].dtype)
            a[pad:Lc], a[Lc:] = cal[k], fc[k]
            out[k] = a
        out["market"] = MARKET_IDS[ALIAS.get(market, market)]
        out["sector"] = int(ser.sector[i])
        out["inst"] = int(ser.vocab[i])
        out["scale"] = TIMESCALE_IDS[scale]
        out["unit"] = self.unit[market, scale]
        out["Lc"] = Lc
        return out


def collate(rows):
    out = {}
    for k in rows[0]:
        if k != "Lc":
            out[k] = torch.as_tensor(np.stack([r[k] for r in rows]))
    return out


def load_config(size):
    return Config.load_yaml(f"{KIT}/configs/dit_{size}.yaml", base=Config())


class PeerBlock(nn.Module):
    """Attention across the stocks of a batch, read and written through each stock's registers."""

    def __init__(self, d, heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, reg):
        h = self.norm(reg.mean(1))[None]
        return reg + self.out(self.attn(h, h, h, need_weights=False)[0][0])[:, None]


class PathDiT(KlineDiT):
    """KlineDiT plus a per-token scale embedding and optional blocks that attend across stocks.

    Both start at zero. Peer blocks run only when peer_on is set, which needs a batch whose
    stocks all share one anchor time.
    """

    def __init__(self, cfg, peers=0):
        super().__init__(cfg)
        self.peer_on = False
        self.pix, self.cls = 0, False
        self.peers = nn.ModuleDict()
        for k in range(peers):
            i = (k + 1) * cfg.n_layers // (peers + 1) - 1
            self.peers[str(i)] = PeerBlock(cfg.d_model)
            self.blocks[i].register_forward_hook(self._peer_hook(i))
        self.tok_scale = nn.Embedding(cfg.scale_slots, cfg.d_model)
        nn.init.zeros_(self.tok_scale.weight)
        self._hook_scale = None
        self.in_proj.register_forward_hook(self._add_scale)

    def _peer_hook(self, i):
        def hook(_m, _i, x):
            if not self.peer_on:
                return x
            n = self.cfg.n_registers
            return torch.cat([self.peers[str(i)](x[:, :n]), x[:, n:]], 1)
        return hook

    def join(self, ctx, z):
        # A canvas target is wider than the numeric history, which then fills its first channels.
        return torch.cat([F.pad(ctx, (0, z.shape[-1] - ctx.shape[-1])), z], 1)

    def _add_scale(self, _m, _i, out):
        if self._hook_scale is None:
            return out
        return out + self.tok_scale(self._hook_scale)

    def forward(self, z, t, cond, Lh, return_aux=False):
        self._hook_scale = cond.get("tok_scale")
        return super().forward(z, t, cond, Lh, return_aux)


@torch.no_grad()
def render(x, mask, unit, H, ma=20):
    """Draw normalized candles as chart images, three pixels per bar, up/down/volume in RGB.

    The price axis spans each window's own low to high, and the bottom fifth is the volume
    panel. Returns images [B,3,H,3L] and the log price span the axis covers.
    """
    x = x.float()
    B, L, _ = x.shape
    raw = decode(x, unit[:, None]) * mask[..., None]
    o, hi, lo, c = candle_path(raw)
    huge = torch.tensor(1e6, device=x.device)
    y0 = torch.where(mask, lo, huge).amin(1, keepdim=True)
    y1 = torch.where(mask, hi, -huge).amax(1, keepdim=True)
    span = (y1 - y0).clamp(min=1e-6)
    V = H // 5
    P = H - V - 1
    py = lambda u: (((u - y0) / span).clamp(0, 1) * (P - 1)).round()
    rows = torch.arange(P, device=x.device)[None, :, None]
    between = lambda a, b: (rows >= py(a)[:, None]) & (rows <= py(b)[:, None]) & mask[:, None]
    wick = between(lo, hi)
    body = between(torch.minimum(o, c), torch.maximum(o, c))
    # Bar columns are side, centre, side: the wick shows only in the centre.
    bars = torch.stack([body, wick | body, body], -1).flatten(2)
    up = (c >= o)[:, None, None, :].repeat_interleave(3, -1)
    keep = mask.float()
    n = keep.cumsum(1)
    s = (c * keep).cumsum(1)
    prev_n = torch.cat([torch.zeros_like(n[:, :ma]), n[:, :-ma]], 1)
    prev_s = torch.cat([torch.zeros_like(s[:, :ma]), s[:, :-ma]], 1)
    avg = (s - prev_s) / (n - prev_n).clamp(min=1)
    line = ((rows == py(avg)[:, None]) & mask[:, None]).float()
    v = torch.exp(raw[..., 4]) * keep
    v = v / v.amax(1, keepdim=True).clamp(min=1e-6)
    tall = (v * (V - 1)).round()
    cols = torch.arange(V, device=x.device)[None, :, None]
    vol = ((cols <= tall[:, None]) & mask[:, None]).float()
    img = torch.zeros(B, 3, H, 3 * L, device=x.device)
    # Row 0 is the top of the image, so both panels flip.
    img[:, 0, :P] = (bars[:, None] & up).float().flip(2)[:, 0]
    img[:, 1, :P] = (bars[:, None] & ~up).float().flip(2)[:, 0]
    img[:, 2, :P, 1::3] = line.flip(1)
    img[:, 2, H - V:, 1::3] = vol.flip(1)
    return img, torch.log(span)


class ChartEncoder(nn.Module):
    """Convolutions over a chart, reduced to one token per candle so calendars stay aligned."""

    def __init__(self, d, H, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 3, stride=(1, 3), padding=(1, 0)), nn.GroupNorm(8, base), nn.GELU(),
            nn.Conv2d(base, base, (3, 5), stride=(2, 1), padding=(1, 2)), nn.GroupNorm(8, base), nn.GELU(),
            nn.Conv2d(base, 2 * base, (3, 5), stride=(2, 1), padding=(1, 2)), nn.GroupNorm(8, 2 * base), nn.GELU(),
            nn.Conv2d(2 * base, 2 * base, (3, 5), stride=(2, 1), padding=(1, 2)), nn.GroupNorm(8, 2 * base), nn.GELU())
        self.proj = nn.Linear(2 * base * (self.net(torch.zeros(1, 3, H, 3)).shape[2]), d)

    def forward(self, img):
        return self.proj(self.net(img).flatten(1, 2).transpose(1, 2))


class ChartDiT(PathDiT):
    """PathDiT that sees history only as chart pixels; numeric context never reaches the blocks."""

    def __init__(self, cfg, peers=0, H=96, pix=0, zmax=4.0, cls=False, fuse=0, base=64, level=0, vlm=""):
        super().__init__(cfg, peers)
        self.H, self.pix, self.zmax, self.cls, self.fuse, self.level, self.vlm = H, pix, zmax, cls, fuse, level, vlm
        if vlm:
            # A frozen vision tower from an open VLM reads the same chart; only this projection trains.
            self.vproj = nn.Sequential(nn.LayerNorm(VLM_DIM[vlm]), nn.Linear(VLM_DIM[vlm], cfg.d_model))
            nn.init.zeros_(self.vproj[1].weight)
            nn.init.zeros_(self.vproj[1].bias)
        self.enc = ChartEncoder(cfg.d_model, H, base)
        self.axis = nn.Linear(1, cfg.d_model)
        self._chart = None
        self._level = None
        if level:
            # The history chart redrawn in each future column's price frame; starts at zero.
            self.lvl = nn.Sequential(nn.Linear(3 * pix, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model))
            nn.init.zeros_(self.lvl[2].weight)
            nn.init.zeros_(self.lvl[2].bias)

    def join(self, ctx, z):
        return torch.cat([F.pad(ctx, (0, z.shape[-1] - ctx.shape[-1])), z], 1)

    def _add_scale(self, m, i, out):
        out = super()._add_scale(m, i, out)
        c = self._chart
        f = out[:, c.shape[1]:]
        if self._level is not None:
            f = f + self._level
        return torch.cat([c + out[:, :c.shape[1]] if self.fuse else c, f], 1)

    def forward(self, z, t, cond, Lh, return_aux=False):
        Lc = z.shape[1] - Lh
        img, span = render(z[:, :Lc, :5], cond["pad_mask"][:, :Lc], cond["unit"], self.H)
        self._chart = self.enc(img.to(self.axis.weight.dtype)) + self.axis(span.to(self.axis.weight.dtype))[:, None]
        if self.vlm:
            self._chart = self._chart + self.vproj(vlm_columns(self.vlm, img, Lc).to(self.axis.weight.dtype))
        if self.level:
            prof = level_profile(z[:, :Lc, :5], cond["pad_mask"][:, :Lc], cond["unit"], Lh, self.pix, self.zmax, hl=self.sig_hl)
            self._level = self.lvl(prof.to(self.axis.weight.dtype))
        if self.fuse:
            # Two streams: pixels, and the same history in numbers rescaled to the window's own volatility.
            r = (vol_unit(z[:, :Lc, :5], cond["pad_mask"][:, :Lc], cond["unit"]) / cond["unit"][:, 1]).clamp(0.2, 5)
            scale = torch.ones_like(z)
            scale[:, :Lc, :4] = r[:, None, None]
            return super().forward(z / scale, t, cond, Lh, return_aux)
        return super().forward(torch.cat([torch.zeros_like(z[:, :Lc]), z[:, Lc:]], 1), t, cond, Lh, return_aux)


class NormDiT(PathDiT):
    """Numeric history with the invariance a chart has: each window is rescaled by its own size.

    The scale it removes returns as one token, as the chart's price span does. mode 1 divides by
    recent volatility, mode 2 by the window's high-low range.
    """

    def __init__(self, cfg, peers=0, mode=1):
        super().__init__(cfg, peers)
        self.mode = mode
        self.axis = nn.Linear(1, cfg.d_model)
        self._span = None

    def _add_scale(self, m, i, out):
        out = super()._add_scale(m, i, out)
        c = self._span
        return torch.cat([out[:, :c.shape[1]] + c, out[:, c.shape[1]:]], 1)

    def forward(self, z, t, cond, Lh, return_aux=False):
        Lc = z.shape[1] - Lh
        ctx, mask, unit = z[:, :Lc], cond["pad_mask"][:, :Lc], cond["unit"]
        with torch.no_grad():
            if self.mode == 1:
                r = vol_unit(ctx, mask, unit) / unit[:, 1]
            else:
                _, hi, lo, _ = candle_path(decode(ctx.float(), unit[:, None]) * mask[..., None])
                r = (hi.amax(1) - lo.amin(1)) / (unit[:, 1] * Lc ** 0.5)
            r = r.clamp(0.2, 5)
        scale = torch.ones_like(z)
        scale[:, :Lc, :4] = r[:, None, None]
        self._span = self.axis(torch.log(r)[:, None].to(self.axis.weight.dtype))[:, None].expand(-1, Lc, -1)
        return super().forward(z / scale, t, cond, Lh, return_aux)


VLM_DIM = {"siglip2": 768}
VLM_PATH = {"siglip2": "google/siglip2-base-patch16-224"}
_VLM = {}


def as_photo(img):
    """Our RGB channel layout to an ordinary chart: white page, green up, red down, blue average and volume."""
    up, dn, aux = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    ink = lambda m, rgb: m * (1 - torch.tensor(rgb, device=img.device)[None, :, None, None])
    return (1 - ink(up, (0.1, 0.6, 0.2)) - ink(dn, (0.85, 0.1, 0.1)) - ink(aux * (1 - up) * (1 - dn), (0.2, 0.3, 0.9))).clamp(0, 1)


@torch.no_grad()
def vlm_columns(name, img, L):
    """Frozen VLM vision features of the chart, pooled down each patch column and spread over L candles."""
    dev = img.device
    if (name, dev) not in _VLM:
        # Weights come from the local cache, so runs work offline.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoModel
        m = AutoModel.from_pretrained(VLM_PATH[name], torch_dtype=torch.bfloat16).vision_model.to(dev).eval()
        _VLM[name, dev] = m
    m = _VLM[name, dev]
    x = F.interpolate(as_photo(img.float()), size=(224, 224), mode="bilinear", align_corners=False)
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    h = m(pixel_values=x).last_hidden_state.float()
    g = int(h.shape[1] ** 0.5)
    cols = h.reshape(h.shape[0], g, g, -1).mean(1)
    return F.interpolate(cols.transpose(1, 2), size=L, mode="linear", align_corners=False).transpose(1, 2)


@torch.no_grad()
def level_profile(ctx, mask, unit, Lh, R, zmax, half=10.0, hl=0.0):
    """The history chart projected onto the price rows of each future canvas column [B,Lh,3R].

    Row r of column h sits at the same log price as the canvas bin it labels, sig * sqrt(h) units
    from the last close. Per row: the share of past candles whose high-low range crosses it, the
    same weighted toward recent bars, and the recency-weighted share of closes inside it.
    """
    raw = decode(ctx.float(), unit[:, None]) * mask[..., None]
    _, hi, lo, c = candle_path(raw)
    last = c[:, -1:]
    hi, lo, c = hi - last, lo - last, c - last
    sig = vol_unit(ctx, mask, unit, hl)
    L = ctx.shape[1]
    keep = mask.float()
    w = torch.exp(-torch.arange(L - 1, -1, -1, device=ctx.device) / half)[None] * keep
    w = w / w.sum(1, keepdim=True).clamp(min=1e-6)
    u = keep / keep.sum(1, keepdim=True).clamp(min=1)
    den = sig[:, None] * horizon(Lh, ctx.device)
    edge = (torch.arange(R + 1, device=ctx.device) / R * 2 * zmax - zmax)
    a = (edge[:-1][None, None] * den[..., None])[..., None]
    b = (edge[1:][None, None] * den[..., None])[..., None]
    cross = ((lo[:, None, None] <= b) & (hi[:, None, None] >= a)).float()
    inside = ((c[:, None, None] >= a) & (c[:, None, None] < b)).float()
    feats = [(cross * u[:, None, None]).sum(-1), (cross * w[:, None, None]).sum(-1), (inside * w[:, None, None]).sum(-1)]
    return torch.cat(feats, -1)


def vol_unit(ctx, mask, unit, hl=0.0):
    """Recent per-bar volatility, the price axis unit of the future canvas; known at inference.

    hl > 0 weights bars by a half-life in bars, as RiskMetrics does, so the unit follows a
    volatility regime as it changes; 0 weighs the window evenly.
    """
    raw = decode(ctx.float(), unit[:, None])
    r = (raw[..., 0] + raw[..., 1]) * mask
    n = mask.sum(1)
    w = mask.float()
    if hl:
        w = w * 0.5 ** (torch.arange(ctx.shape[1] - 1, -1, -1, device=ctx.device) / hl)[None]
    v = ((w * r * r).sum(1) / w.sum(1).clamp(min=1e-9)).sqrt()
    return torch.where(n >= 5, v, unit[:, 1]).clamp(min=1e-5)


def rebin(p, ratio, zmax):
    """Canvas bin probabilities [B,Lh,R] drawn in one volatility unit, re-read on bins of another.

    ratio is the drawing unit over the reading unit per row; within a bin mass is uniform.
    """
    B, Lh, R = p.shape
    e = torch.linspace(-zmax, zmax, R + 1, device=p.device)
    src = (e[None] * ratio[:, None]).float()
    cdf = torch.cat([torch.zeros_like(p[..., :1]), p.float().cumsum(-1)], -1)
    q = e[None].expand(B, -1).contiguous()
    i = torch.searchsorted(src.contiguous(), q).clamp(1, R)
    lo, hi = src.gather(1, i - 1), src.gather(1, i)
    f = ((q - lo) / (hi - lo)).clamp(0, 1)
    f = torch.where(q < src[:, :1], torch.zeros_like(f), torch.where(q > src[:, -1:], torch.ones_like(f), f))
    i, f = i[:, None].expand(-1, Lh, -1), f[:, None].expand(-1, Lh, -1)
    c = cdf.gather(2, i - 1) * (1 - f) + cdf.gather(2, i) * f
    c[..., 0], c[..., -1] = 0, 1
    return c.diff(dim=-1).clamp(min=0)


def horizon(Lh, device):
    return torch.arange(1, Lh + 1, device=device, dtype=torch.float32).sqrt()


def pix_target(fut, sig, unit, R, zmax):
    """Future candles as a canvas [B,Lh,2R] in {-1,1}: the close's bin, then the bins the bar spans.

    Row positions are returns in units of sig * sqrt(h). The mean of a {0,1} pixel is the
    probability it is lit, which is why the one-step mean of this canvas is a distribution.
    """
    o, hi, lo, c = candle_path(decode(fut.float(), unit[:, None]))
    den = sig[:, None] * horizon(fut.shape[1], fut.device)
    row = lambda u: (((u / den + zmax) / (2 * zmax) * R).floor().clamp(0, R - 1))[..., None]
    rows = torch.arange(R, device=fut.device)[None, None]
    close = rows == row(c)
    span = (rows >= row(lo)) & (rows <= row(hi))
    return torch.cat([close, span], -1).float() * 2 - 1


def pix_expect(m, sig, R, zmax):
    """Expected cumulative log return per future bar from a mean canvas."""
    p = ((m[..., :R].float() + 1) / 2).clamp(min=0) + 1e-6
    p = p / p.sum(-1, keepdim=True)
    c = (torch.arange(R, device=m.device) + 0.5) / R * 2 * zmax - zmax
    return (p * c).sum(-1) * sig[:, None] * horizon(m.shape[1], m.device)


def pix_probs(logits, R):
    """Classifier output on the canvas's {-1,1} scale: softmax over close bins, sigmoid over span bins."""
    return torch.cat([logits[..., :R].softmax(-1), logits[..., R:].sigmoid()], -1) * 2 - 1


def pix_y(c, sig, zmax):
    """Cumulative return in units of sig * sqrt(h), the channel that keeps what the rows quantize."""
    return (c / (sig[:, None] * horizon(c.shape[1], c.device))).clamp(-2 * zmax, 2 * zmax)


def pix_numeric(m, sig, unit, R, zmax):
    """A canvas in the normalized feature layout, so the numeric pipelines read it unchanged."""
    e = m[..., 2 * R] * sig[:, None] * horizon(m.shape[1], m.device) if m.shape[-1] > 2 * R else pix_expect(m, sig, R, zmax)
    r = torch.diff(e, dim=1, prepend=torch.zeros_like(e[:, :1]))
    out = torch.zeros(*e.shape, 5, device=m.device)
    out[..., 1] = 6 * torch.tanh(r / (6 * unit[:, 1, None]))
    return out


def build_model(size, n_layers=None, d_model=None, like=None, peers=0, chart=0, pix=0, zmax=4.0, cls=False, sub=False, yraw=False, norm=0, fuse=0, base=64, level=0, vlm="", sig_hl=0.0):
    """like: a KiT checkpoint whose architecture to copy, needed since its vocabulary differs from the yaml."""
    cfg = Config.from_dict(torch.load(like, map_location="cpu", weights_only=False)["config"]) if like else load_config(size)
    m = cfg.model
    m.attn_backend = "sdpa"
    m.inst_vocab = 8192  # The yaml says 4096, but the released instrument ids run past it.
    if n_layers:
        m.n_layers = n_layers
    if d_model:
        m.ffn_hidden = int(m.ffn_hidden * d_model / m.d_model)
        m.d_model = d_model
    if pix:
        m.in_features = 2 * pix + int(sub)
    net = ChartDiT(m, peers, chart, pix, zmax, cls, fuse, base, level, vlm) if chart else NormDiT(m, peers, norm) if norm else PathDiT(m, peers)
    # The canvas head is set on every class, so a numeric-history model can draw the same canvas.
    net.pix, net.zmax, net.cls, net.yraw, net.sig_hl = pix, zmax, cls, yraw, sig_hl
    return net, cfg


def load_kit_weights(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ema = ck["ema"]["shadow"] if "shadow" in ck["ema"] else ck["ema"]
    miss, extra = model.load_state_dict({k: v.float() for k, v in ema.items()}, strict=False)
    assert not extra, extra
    assert all(k.startswith("tok_scale") for k in miss), miss
    return ck.get("step")


def scale_tensor(stats, market, scale, device):
    market = ALIAS.get(market, market)
    return torch.tensor([stats[stats_key(market, scale, n)] for n in FEATURES], device=device)


def decode(y, scale, c=6.0):
    """Normalized features to raw log ratios; the torch form of KiT's denormalize."""
    r = (y / c).clamp(-(1 - 1e-4), 1 - 1e-4)
    return torch.atanh(r) * c * scale


def candle_path(f):
    """Raw features [...,L,4+] to per-bar open/high/low/close, relative to the anchor close."""
    gap, body, up, dn = f[..., 0], f[..., 1], f[..., 2].clamp(min=0), f[..., 3].clamp(min=0)
    c = (gap + body).cumsum(-1)
    o = c - body
    hi = torch.maximum(o, c) + up
    lo = torch.minimum(o, c) - dn
    return o, hi, lo, c


def measure(o, hi, lo, c, w):
    """The candle operator: O=first, H=max, L=min, C=last over non-overlapping w-bar blocks."""
    n = c.shape[-1] // w * w
    sh = c.shape[:-1] + (-1, w)
    return (o[..., :n].reshape(sh)[..., 0], hi[..., :n].reshape(sh).amax(-1),
            lo[..., :n].reshape(sh).amin(-1), c[..., :n].reshape(sh)[..., -1])


def bars_day(market, scale):
    return bars_per_day(ALIAS.get(market, market), scale)
