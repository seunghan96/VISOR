"""Zero-shot LLM ranker (LAFP-style ranking step): a text LLM ranks the SAME 32 numeric candidates of a reported
VISOR run given the same text, on the same test rows, and names the index of the most plausible future (greedy).

The prompt is the number-input prompt of visor/selector.py (--input num: history and candidates on one shared 0-99 scale, the same
text block), with the marker tokens removed and an instruction to answer with the candidate number only. The LLM is
the language model of Qwen2.5-VL-3B-Instruct, used without any image and without training. Unparsable answers fall back
to the medoid (counted). Usage: python baselines/lafp_rank.py --ref <selector tag> --split te|te_g17 [--shard i/n]
(<selector tag>: a finished selector run with results/<tag>.json and selector_runs/<tag>_<split>_scores.npz)
Writes lafp_runs/<ref>_<split>[_shard].npz with rows, pick (candidate id in 0..31), fail flags.
"""
import argparse, json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import visor.selector as V  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--ref", required=True)
p.add_argument("--split", default="te")
p.add_argument("--shard", default="0/1")
p.add_argument("--bs", type=int, default=8)
p.add_argument("--tag", default="")
q = p.parse_args()

a = argparse.Namespace(**json.load(open(f"results/{q.ref}.json"))["args"])
for k, v in dict(data="fnspid", anon=0, news_days=5, news_clean=0, max_news=8, n_summary=2, mode="candle", text="news",
                 series_desc="Time series: {}", side=V.SIDE[0]).items():
    if not hasattr(a, k):
        setattr(a, k, v)
a.input = "num"
V.SIDE[0] = getattr(a, "side", V.SIDE[0])
TE = {n: V.load(pt) for n, pt in (x.split("=") for x in a.test.split(";") if x)}
D = TE[q.split]
z = np.load(f"selector_runs/{q.ref}_{q.split}_scores.npz")
rows = z["rows"]
si, sn = map(int, q.shard.split("/"))
part = np.array_split(np.arange(len(rows)), sn)[si]
K = 32
if a.data == "tmmd":
    prices, news = V.TPrices(a.hist), V.TNews()
else:
    import pandas as pd
    codes = set(D["code"][rows])
    prices = V.Prices(codes, a.hist)
    td = pd.to_datetime(pd.read_csv(f"{V.MEM}/1d/trade_dates.csv").date.astype(str)).values.astype("datetime64[D]")
    news = V.News(codes, td, a.news_days, a.news_clean, a.anon)
feeder = V.Feeder(prices, news, a)
tok = feeder.proc.tokenizer
from transformers import AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained(a.vlm, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

rng = np.random.default_rng(0)
orders = [rng.permutation(K) for _ in range(len(rows))]   # fixed random panel order per row, as in VISOR's evaluation
INSTR = "\nAnswer with the number of the most plausible candidate only."


def prompt(i, kk):
    t = feeder.prompt_num(D, i, kk, i)
    t = t.replace(V.MARK, "")
    j = t.rfind("<|im_end|>")
    return t[:j] + INSTR + t[j:]


pick = np.full(len(part), -1, np.int64)
fail = np.zeros(len(part), bool)
c = D["draws"][rows[part], :K, :, 2].astype(np.float32)
med = np.abs(c[:, :, None] - c[:, None]).mean(-1).sum(-1).argmin(1)
for s in range(0, len(part), q.bs):
    js = part[s:s + q.bs]
    texts = [prompt(rows[j], orders[j]) for j in js]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=6, do_sample=False)
    ans = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    for b, (j, t) in enumerate(zip(js, ans)):
        m = re.search(r"\d+", t)
        n = int(m.group(0)) if m else -1
        loc = s + b
        if 1 <= n <= K:
            pick[loc] = orders[j][n - 1]
        else:
            pick[loc], fail[loc] = med[loc], True
    if s % (q.bs * 25) == 0:
        print(f"{s}/{len(part)} fail {fail[:s + len(js)].mean():.3f} ex {ans[0]!r}", flush=True)
os.makedirs("lafp_runs", exist_ok=True)
suf = "" if sn == 1 else f"_{si}of{sn}"
np.savez(f"lafp_runs/{q.ref}_{q.split}{suf}.npz", rows=rows[part], pick=pick, fail=fail)
print("done", len(part), "fail rate", fail.mean(), flush=True)
json.dump({"ref": q.ref, "split": q.split, "shard": q.shard, "n": int(len(part)), "fail": float(fail.mean())},
          open(f"results/{q.tag or 'lafp_' + q.ref + '_' + q.split + suf}.json", "w"))
