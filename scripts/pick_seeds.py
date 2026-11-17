"""Seed protocol of the paper: five selector seeds, report the mean over the three with the lowest validation target.

  python scripts/pick_seeds.py <train tag> [--test_tag <tag of the test-only run>] [--split te] [--n 3]
e.g. python scripts/pick_seeds.py vt_3b_w_shape0.1 --test_tag te3_vt_3b_w_shape0.1 --split te_g17   (FNSPID)
     python scripts/pick_seeds.py tmmdP_A_ftCPpri_Energy                                            (test in the same run)
Seed 0 has no suffix, seed s the suffix _s<s>. val_best (results/<tag>.json) picks the seeds; the test metrics of the
selector and of the medoid rule on the same rows are averaged over them.
"""
import argparse
import json
import os

import numpy as np


def main():
    q = argparse.ArgumentParser()
    q.add_argument("tag")
    q.add_argument("--test_tag", default="")
    q.add_argument("--split", default="te")
    q.add_argument("--n", type=int, default=3)
    q.add_argument("--seeds", default="0,1,2,3,4")
    a = q.parse_args()
    runs = []
    for s in map(int, a.seeds.split(",")):
        sfx = "" if s == 0 else f"_s{s}"
        f = f"results/{a.tag}{sfx}.json"
        if not os.path.exists(f):
            print("missing", f)
            continue
        val = json.load(open(f))["val_best"]
        te = json.load(open(f"results/{a.test_tag}{sfx}.json" if a.test_tag else f))[a.split]
        runs.append((val, s, te))
    runs.sort(key=lambda r: r[0])
    pick = runs[:a.n]
    print("seeds by val target:", [(s, round(v, 4)) for v, s, _ in runs], "-> picked", [s for _, s, _ in pick])
    keys = [k for k in pick[0][2] if isinstance(pick[0][2][k], dict) and k.split("/")[-1] in ("selector", "medoid")]
    out = {k: {m: float(np.mean([r[2][k][m] for r in pick])) for m in pick[0][2][k]} for k in keys}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
