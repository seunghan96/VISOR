"""Per-dataset candidate pools from the Time-MMD predictor dumps, and the predictor setting chosen by cross-validation.

flow_mlp.py writes the windows of all eight Time-MMD datasets into one file per split (sel_dump/<src>_{tr,va,te}.npz).
The selector of each dataset reads only that dataset's rows:

  python -m visor.pools slice <src> <dataset> <out>
        -> sel_dump/<out>_<dataset>_{tr,va,te}.npz (rows of <dataset> in <src>, checked against <out>_<dataset> if it
           exists with a different source, e.g. tmmdC_<dataset> built from the default predictor setting)

  python -m visor.pools cv <dataset> <current> <config> [<config> ...]
        Predictor setting by cross-validation: every config was built with flow_mlp.py --per_dom 1 --oof 5
        --out <config> (out-of-fold train draws). The medoid MAE of each config is computed over the out-of-fold
        train rows plus the validation rows (test is never read). The best config replaces <current> only if a paired
        one-sided t-test gives p < 0.05 / (number of other configs). Configs built with --extra 1 are left out, as
        their folds share the extra columns of the same period.
"""
import json
import sys

import numpy as np
from scipy import stats


def part(src, d, s):
    z = dict(np.load(f"sel_dump/{src}_{s}.npz"))
    return {k: v[z["code"] == d] for k, v in z.items()}


def slice_(src, d, out):
    import os
    for s in ("tr", "va", "te"):
        z = part(src, d, s)
        assert len(z["day"]), (src, d, s)
        ref = f"sel_dump/{out}_{d}_{s}.npz"
        if os.path.exists(ref):
            r = dict(np.load(ref))
            for k in ("day", "code", "sig", "y"):
                assert np.array_equal(r[k], z[k]), (d, s, k)
        np.savez(f"sel_dump/{out}_{d}_{s}.npz", **z)
        print(out, d, s, len(z["day"]), "rows", flush=True)


def row_err(c, d):
    """Per-row medoid MAE (close path) of config c on dataset d over the train (out-of-fold) and validation rows."""
    e = []
    for s in ("tr", "va"):
        z = part(c, d, s)
        Dr, y = z["draws"][..., 2].astype(float), z["y"][..., 2].astype(float)
        med = Dr[np.arange(len(Dr)), np.abs(Dr[:, :, None] - Dr[:, None]).mean(-1).sum(-1).argmin(1)]
        e.append(np.abs(med - y).mean(1))
    return np.concatenate(e)


def cv(d, cur, configs):
    ok = sorted(set(configs) | {cur})
    E = {c: row_err(c, d) for c in ok}
    best = min(E, key=lambda c: E[c].mean())
    p = float(stats.ttest_rel(E[best], E[cur], alternative="less").pvalue) if best != cur else 1.0
    adopt = best != cur and p < 0.05 / max(len(ok) - 1, 1)
    res = dict(dataset=d, current=cur, current_cv=float(E[cur].mean()), best=best, best_cv=float(E[best].mean()), p=p,
               adopt=bool(adopt), pick=best if adopt else cur, n=len(E[cur]),
               top5=sorted(((float(E[c].mean()), c) for c in E))[:5])
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "slice":
        slice_(*sys.argv[2:5])
    elif sys.argv[1] == "cv":
        cv(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        raise SystemExit(__doc__)
