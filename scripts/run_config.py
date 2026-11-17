"""Print (default) or run the commands of a configs/*.json file, in order, from the repository root.

  python scripts/run_config.py configs/fnspid.json                      print every step
  python scripts/run_config.py configs/fnspid.json --step selector_train --seeds 0 --run
  python scripts/run_config.py configs/tmmd_Energy.json --skip_common   leave out the steps shared by all Time-MMD datasets

Placeholders: {p} part index (and {s100} = 100 + p, the seed of a val / test part), {seed} the selector seed and {sfx}
its tag suffix ("" for seed 0, "_s<seed>" otherwise).
"""
import argparse
import json
import subprocess
import sys


def expand(step, seeds):
    if "loop" in step:
        for p in step["loop"]["p"]:
            yield step["cmd"].replace("{p}", str(p)).replace("{s100}", str(100 + p))
    elif "seeds" in step:
        for s in seeds if seeds is not None else step["seeds"]:
            yield step["cmd"].replace("{seed}", str(s)).replace("{sfx}", "" if s == 0 else f"_s{s}").replace("{{}}", "{}")
    else:
        yield step["cmd"]


def main():
    q = argparse.ArgumentParser()
    q.add_argument("config")
    q.add_argument("--step", default="", help="Comma-separated step names (default: all)")
    q.add_argument("--seeds", default="", help="Comma-separated selector seeds (default: the config's five)")
    q.add_argument("--skip_common", action="store_true")
    q.add_argument("--run", action="store_true", help="Run the commands instead of printing them")
    a = q.parse_args()
    cfg = json.load(open(a.config))
    want = set(a.step.split(",")) if a.step else None
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    for st in cfg["steps"]:
        if (want and st["name"] not in want) or (a.skip_common and st.get("common")):
            continue
        for cmd in expand(st, seeds):
            print(f"# {st['name']}\n{cmd}", flush=True)
            if a.run:
                r = subprocess.run(cmd, shell=True)
                if r.returncode:
                    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
