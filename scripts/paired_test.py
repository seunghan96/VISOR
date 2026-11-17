"""Paired t-test, clustered by forecast time, between two selector runs (or a run and the medoid rule) on the same rows.

  python scripts/paired_test.py <dump glob> <scores_a.npz> [<scores_b.npz> | medoid] [--k 32] [--metric mae]
e.g. python scripts/paired_test.py "sel_dump/fsel_fnspid_te_g17_p*.npz" selector_runs/te3_vt_3b_w_shape0.1_te_g17_scores.npz medoid
Score files are the selector's selector_runs/<tag>_<split>_scores.npz (rows, scores). t < 0: a has the lower error.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visor.selector import load  # noqa: E402
from visor.significance import medoid, paired, path_metrics  # noqa: E402


def main():
    q = argparse.ArgumentParser()
    q.add_argument("dump")
    q.add_argument("a")
    q.add_argument("b", nargs="?", default="medoid")
    q.add_argument("--k", type=int, default=32)
    q.add_argument("--metric", default="mae")
    a = q.parse_args()
    D = load(a.dump)
    za = np.load(a.a)
    rows = za["rows"]
    pa = za["scores"].argmax(1)
    if a.b == "medoid":
        pb = medoid(D, rows, a.k)
    else:
        zb = np.load(a.b)
        assert np.array_equal(zb["rows"], rows), "score files cover different rows"
        pb = zb["scores"].argmax(1)
    ma, fa = path_metrics(D, rows, pa, a.k)
    mb, fb = path_metrics(D, rows, pb, a.k)
    for m in ([a.metric] if a.metric != "all" else ma):
        print(f"{m}: a {ma[m].mean():.4f}  b {mb[m].mean():.4f}  t {paired(ma[m], mb[m], D['day'][rows]):+.2f}")
    print(f"FD shape: a {fa:.3f}  b {fb:.3f}")


if __name__ == "__main__":
    main()
