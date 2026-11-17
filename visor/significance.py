"""Per-row metrics of one chosen path and paired t-tests clustered by forecast time.

A paired t-statistic is computed on the per-row differences of two methods averaged within each forecast time
(daymean), so the common move of all series at one step counts once; |t| > 2 is called significant. On Time-MMD each
dataset is one series with one row per forecast time, and the test reduces to a paired t-test over windows.

scripts/paired_test.py runs it on two saved selector score files. In code:
    m_a, _ = path_metrics(D, rows, pick_a, K); m_b, _ = path_metrics(D, rows, pick_b, K)
    t = paired(m_a["mae"], m_b["mae"], D["day"][rows])
"""
import numpy as np

from visor.metrics import features, frechet


def daymean(x, day):
    _, inv = np.unique(day, return_inverse=True)
    return np.bincount(inv, x) / np.bincount(inv)


def tstat(x):
    return float(x.mean() / (x.std() + 1e-12) * np.sqrt(len(x)))


def paired(a, b, day):
    """t-statistic of a - b (per-row metric arrays of two methods on the same rows), clustered by forecast time.
    Negative values mean a has the lower error."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    f = np.isfinite(d)
    return tstat(daymean(d[f], np.asarray(day)[f]))


def path_metrics(D, rows, pick, K):
    """Per-row metrics of the chosen candidate (high, low, close in path units) and the FD of its features.
    mae, mse: close path; end: endpoint error; hl: MAE of the high and low paths; volerr, rangeerr: volatility and
    high-low range error; dderr, runuperr: largest drop and largest rise error."""
    dr = D["draws"][rows, :K].astype(np.float32)
    y = D["y"][rows].astype(np.float32)
    p = dr[np.arange(len(rows)), pick]
    fp, fy = features(p), features(y)
    m = dict(mae=np.abs(p[..., 2] - y[..., 2]).mean(1), mse=((p[..., 2] - y[..., 2]) ** 2).mean(1),
             end=np.abs(p[:, -1, 2] - y[:, -1, 2]), hl=np.abs(p[..., :2] - y[..., :2]).mean((1, 2)),
             volerr=np.abs(fp[:, 0] - fy[:, 0]), rangeerr=np.abs(fp[:, 6] - fy[:, 6]),
             dderr=np.abs(fp[:, 2] - fy[:, 2]), runuperr=np.abs(fp[:, 3] - fy[:, 3]))
    return m, frechet(fp, fy)


def medoid(D, rows, K):
    """Index of the candidate closest (mean absolute close distance) to all others."""
    c = D["draws"][rows, :K, :, 2].astype(np.float32)
    return np.abs(c[:, :, None] - c[:, None]).mean(-1).sum(-1).argmin(1)
