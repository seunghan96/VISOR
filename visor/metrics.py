"""Path metrics, training-free selection rules, and the selector's evaluator.

All paths are (high, low, close) arrays [N, H, 3] in the path unit of the paper, (x_{t+h} - x_t) / (sigma sqrt(h)).
evaluate() scores the selector's pick against the per-step median, a random candidate, the training-free rules
(medoid, mode, closest to median / mean, maximum likelihood, ...) and the best candidate in hindsight (oracle).
report() prints MAE, MSE, volatility and range error with a paired t-statistic against the medoid on per-day means
(Newey-West, 2 lags), and the Frechet distance (FD) of seven path features to those of the true paths.
"""
import numpy as np
import torch.nn.functional as F


def nwt(d, lags=2):
    """1 / standard error of the mean of d with a Newey-West variance (lags lags)."""
    d = d - d.mean()
    v = (d * d).mean()
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (d[k:] * d[:-k]).mean()
    return np.sqrt(len(d) / max(v, 1e-12))


def daily(x, day):
    """Mean of x within each forecast time."""
    u, inv = np.unique(day, return_inverse=True)
    return np.bincount(inv, x) / np.bincount(inv)


def objective(logit, d, a):
    """Selector loss. reg (VISOR, Eq. "loss"): the scores match the negative distances d up to a constant per window,
    so the negative score estimates the expected distance. ce: cross-entropy to softmax(-d / tau)."""
    if a.obj == "reg":
        e = -logit - d
        return ((e - e.mean(-1, keepdim=True)) ** 2).mean()
    target = F.softmax(-d / a.tau, -1)
    return -(target * F.log_softmax(logit, -1)).sum(-1).mean()


def features(p):
    """Shape of (high, low, close) paths [N,Lh,3] given in units of sig * sqrt(h), measured in sig per day."""
    q = p * np.sqrt(np.arange(1, p.shape[1] + 1))[None, :, None]
    c = q[..., 2]
    r = np.diff(np.c_[np.zeros(len(c)), c], axis=1)
    run = np.maximum.accumulate(np.c_[np.zeros(len(c)), c], 1)[:, 1:]
    low = np.minimum.accumulate(np.c_[np.zeros(len(c)), c], 1)[:, 1:]
    rc = r - r.mean(1, keepdims=True)
    ac = (rc[:, 1:] * rc[:, :-1]).mean(1) / np.maximum((rc ** 2).mean(1), 1e-6)
    return np.stack([r.std(1), np.abs(r).mean(1), (run - c).max(1), (c - low).max(1), c[:, -1], ac,
                     (q[..., 0] - q[..., 1]).mean(1)], 1)


FEATS_N = ("vol", "abs ret", "max dd", "max runup", "10d ret", "autocorr", "range")


def frechet(a, b):
    """Frechet distance between Gaussian fits of two feature sets, after scaling by the truth's spread."""
    s = b.std(0) + 1e-6
    a, b = a / s, b / s
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    w, v = np.linalg.eigh(ca)
    sa = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    w2 = np.linalg.eigvalsh(sa @ cb @ sa)
    return float(((ma - mb) ** 2).sum() + np.trace(ca) + np.trace(cb) - 2 * np.sqrt(np.clip(w2, 0, None)).sum())


def rules(dr, chunk=2048):
    """Training-free picks from K draws [N,K,Lh,3]: index per row for each rule."""
    N, K = dr.shape[:2]
    out = {k: np.zeros(N, int) for k in ("medoid", "medoid_hlc", "mode", "mode_narrow", "near_median", "near_mean",
                                         "marglik", "typical25", "typical50")}
    for i in range(0, N, chunk):
        x = dr[i:i + chunk]
        c = x[..., 2]
        n = len(c)
        d1 = np.abs(c[:, :, None] - c[:, None]).mean(-1)
        d2 = ((c[:, :, None] - c[:, None]) ** 2).mean(-1)
        d3 = np.abs(x[:, :, None] - x[:, None]).mean((-1, -2))
        out["medoid"][i:i + n] = d1.sum(-1).argmin(1)
        out["medoid_hlc"][i:i + n] = d3.sum(-1).argmin(1)
        bw = np.median(d2.reshape(n, -1), 1)[:, None, None] + 1e-6
        out["mode"][i:i + n] = np.exp(-d2 / (2 * bw)).sum(-1).argmax(1)
        out["mode_narrow"][i:i + n] = np.exp(-d2 / (0.5 * bw)).sum(-1).argmax(1)
        out["near_median"][i:i + n] = np.abs(c - np.median(c, 1, keepdims=True)).mean(-1).argmin(1)
        out["near_mean"][i:i + n] = ((c - c.mean(1, keepdims=True)) ** 2).mean(-1).argmin(1)
        g = np.exp(-(c[:, :, None] - c[:, None]) ** 2 / (2 * 0.3 ** 2)).mean(2)
        out["marglik"][i:i + n] = np.log(g + 1e-9).sum(-1).argmax(1)
        # Central in location, then the draw whose daily volatility is closest to the draws' typical one.
        vol = features(x.reshape(-1, *x.shape[2:]))[:, 0].reshape(n, K)
        rank = d1.sum(-1).argsort(1).argsort(1)
        for q, name in ((0.25, "typical25"), (0.5, "typical50")):
            gap = np.abs(vol - np.median(vol, 1, keepdims=True)) + 1e9 * (rank >= max(1, int(q * K)))
            out[name][i:i + n] = gap.argmin(1)
    return out


def evaluate(D, pick, seed=0):
    """Ways to turn K draws into one path: accuracy per row, and per-row shape features for realism."""
    dr = D["draws"].astype(np.float32)
    y = D["y"].astype(np.float32)
    N, K = dr.shape[:2]
    c = dr[..., 2]
    err = np.abs(c - y[:, None, :, 2]).mean(-1)
    at = lambda i: dr[np.arange(N), i]
    paths = {"median": np.median(dr, 1), "random": at(np.random.default_rng(seed).integers(K, size=N))}
    paths |= {k: at(v) for k, v in rules(dr).items()}
    paths |= {"selector": at(pick), "oracle": at(err.argmin(1))}
    tr = y[..., 2]
    ft = features(y)
    rows = {"truth": dict(feat=ft)}
    for n, p in paths.items():
        f = features(p)
        rows[n] = dict(mae=np.abs(p[..., 2] - tr).mean(1), mse=((p[..., 2] - tr) ** 2).mean(1),
                       end=np.abs(p[:, -1, 2] - tr[:, -1]), hl=np.abs(p[..., :2] - y[..., :2]).mean((1, 2)),
                       volerr=np.abs(f[:, 0] - ft[:, 0]), rangeerr=np.abs(f[:, 6] - ft[:, 6]), feat=f)
    return rows


def report(D, rows, ref="medoid"):
    """Accuracy with paired daily t against ref, then realism: Frechet distance of path shapes to the truth's."""
    day = D["day"]
    res = {}
    ms = ("mae", "mse", "end", "hl", "volerr", "rangeerr")
    for mk, name in enumerate(("us", "ashare", "crypto")):
        sel = D["mk"] == mk
        if not sel.any():
            continue
        ft = rows["truth"]["feat"][sel]
        print(f"\n{name}: {sel.sum()} rows")
        print(f"{'path':10s}" + "".join(f"{m:>16s}" for m in ("MAE", "MSE", "endpoint", "HL", "vol err", "range err")) + f"{'FD shape':>10s}"
              + "".join(f"{m:>10s}" for m in FEATS_N))
        print(f"{'truth':10s}" + " " * 96 + f"{0:10.3f}" + "".join(f"{v:10.3f}" for v in ft.mean(0)))
        for n, r in rows.items():
            if n == "truth":
                continue
            cells = []
            for m in ms:
                v = r[m][sel]
                if n != ref:
                    d = daily(v - rows[ref][m][sel], day[sel])
                    cells.append(f"{v.mean():9.4f} t{d.mean() * nwt(d):+5.1f}")
                else:
                    cells.append(f"{v.mean():9.4f}       ")
            fd = frechet(r["feat"][sel], ft)
            print(f"{n:10s}" + "".join(f"{c:>16s}" for c in cells) + f"{fd:10.3f}" + "".join(f"{v:10.3f}" for v in r["feat"][sel].mean(0)))
            res[f"{name}/{n}"] = {m: float(r[m][sel].mean()) for m in ms} | {"fd": fd}
    return res
