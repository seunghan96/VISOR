"""Released time-series foundation models, run as published, on the rows scripts/data/export_rows.py exported.

Each model sees the same 90 daily closes as every other model and forecasts 10 closes with its
released weights and default inference settings; nothing is tuned. The output is a set of
cumulative log-return paths [N, K, 10] for px_tsfm_score.py.

Quantile models give marginals only. Their paths are drawn as inverse CDFs on an even grid of
levels, one level per path across the horizon, and the forecast is flat beyond its outer quantiles.
Point models give one path.

Run it with the python of an environment that has the model's package (chronos-forecasting, timesfm, uni2ts,
toto-ts); see README. Example: python baselines/px_tsfm.py --family chronos --model amazon/chronos-t5-base
--market fnspid_te23 --tag fxt_chronos_t5b   (Time-MMD: --market tmmd_te --lh 12 --tag tmfxt_<name>; FinTexTS:
--market fintexts_te --lh 3 --tag ftxfxt_<name>). Output: px_tsfm_out/<tag>.npz, scored by baselines/score_baselines.py.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

LH = 10


def from_quantiles(levels, q, K=256):
    """Samples on an even level grid from quantiles q [N, Lh, Q] at levels [Q]."""
    u = (np.arange(K) + 0.5) / K
    q = np.sort(q, -1)
    flat = q.reshape(-1, q.shape[-1])
    s = np.stack([np.interp(u, levels, r) for r in flat]).reshape(q.shape[0], q.shape[1], K)
    return s.transpose(0, 2, 1)


def run_chronos(name, ctx, dev):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(name, device_map=dev, torch_dtype=torch.bfloat16 if "t5" in name else torch.float32)
    out = []
    # Batch size only bounds memory: the sampler draws 100 paths per series.
    bs = 16 if "t5" in name else 256
    for j in range(0, len(ctx), bs):
        x = torch.tensor(ctx[j:j + bs], dtype=torch.float32)
        if "t5" in name:
            # Sampling model: the pipeline's own sampler with its default temperature, top_k and top_p.
            s = pipe.predict(x, prediction_length=LH, num_samples=100)
            out.append(s.float().cpu().numpy())
        else:
            lv = [0.01, 0.05] + [i / 10 for i in range(1, 10)] + [0.95, 0.99] if "chronos-2" in name else [i / 10 for i in range(1, 10)]
            if "chronos-2" in name:
                # Chronos-2 takes (series, variates, time) and returns one (variates, Lh, Q) tensor per series.
                q, _ = pipe.predict_quantiles(x[:, None], prediction_length=LH, quantile_levels=lv)
                q = torch.stack([t[0] for t in q]) if isinstance(q, list) else q[:, 0]
            else:
                q, _ = pipe.predict_quantiles(x, prediction_length=LH, quantile_levels=lv)
            out.append(from_quantiles(np.array(lv), q.float().cpu().numpy()))
    return np.concatenate(out), "samples" if "t5" in name else "quantiles"


def run_timesfm(name, ctx, dev):
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(name)
    m.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256, normalize_inputs=True, use_continuous_quantile_head=True,
                                     force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
    _, q = m.forecast(horizon=LH, inputs=[c for c in ctx])
    # Channel 0 is the mean, then the 10th to 90th percentiles.
    return from_quantiles(np.arange(1, 10) / 10, np.asarray(q)[:, :LH, 1:]), "quantiles"


def run_moirai(name, ctx, dev):
    if "moirai-2" in name:
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        mod = Moirai2Forecast(module=Moirai2Module.from_pretrained(name), prediction_length=LH, context_length=ctx.shape[1],
                              target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    else:
        # Moirai 1.x through its README's GluonTS predictor, which pads the context for patch_size="auto".
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        mod = MoiraiForecast(module=MoiraiModule.from_pretrained(name), prediction_length=LH, context_length=ctx.shape[1],
                             patch_size="auto", num_samples=100, target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        pred = mod.create_predictor(batch_size=128, device=dev)
        ds = ListDataset([{"start": pd.Period("2000-01-03", freq="B"), "target": c} for c in ctx], freq="B")
        return np.stack([f.samples for f in pred.predict(ds)]), "samples"
    mod = mod.to(dev).eval()
    out = []
    for j in range(0, len(ctx), 128):
        x = torch.tensor(ctx[j:j + 128], dtype=torch.float32, device=dev)[..., None]
        obs = torch.ones_like(x, dtype=torch.bool)
        pad = torch.zeros(x.shape[:2], dtype=torch.bool, device=dev)
        with torch.no_grad():
            if "moirai-2" in name:
                q = mod(past_target=x, past_observed_target=obs, past_is_pad=pad)
                out.append(from_quantiles(np.array(mod.module.quantile_levels), q.float().cpu().numpy().transpose(0, 2, 1)))
            else:
                out.append(mod(past_target=x, past_observed_target=obs, past_is_pad=pad).float().cpu().numpy())
    return np.concatenate(out), "quantiles" if "moirai-2" in name else "samples"


def run_hf(name, ctx, dev):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.float32).to(dev).eval()
    out = []
    for j in range(0, len(ctx), 256):
        x = torch.tensor(ctx[j:j + 256], dtype=torch.float32, device=dev)
        if "sundial" in name.lower():
            # Sundial's documented usage: raw series in, generated samples out; its default is 20.
            g = m.generate(x, max_new_tokens=LH, num_samples=100)
            out.append(g.float().cpu().numpy()[..., -LH:])
        else:
            # Time-MoE's documented usage: the caller normalizes, the model returns one point path.
            mu, sd = x.mean(-1, keepdim=True), x.std(-1, keepdim=True)
            g = m.generate((x - mu) / sd, max_new_tokens=LH)[:, -LH:]
            out.append((g * sd + mu).float().cpu().numpy()[:, None])
    return np.concatenate(out), "samples" if "sundial" in name.lower() else "point"


def run_toto(name, ctx, dev):
    from toto.data.util.dataset import MaskedTimeseries
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto
    m = Toto.from_pretrained(name).to(dev)
    f = TotoForecaster(m.model)
    out = []
    for j in range(0, len(ctx), 64):
        x = torch.tensor(ctx[j:j + 64], dtype=torch.float32, device=dev)[:, None]
        ts = torch.zeros_like(x)
        mt = MaskedTimeseries(series=x, padding_mask=torch.ones_like(x, dtype=torch.bool), id_mask=torch.zeros_like(x),
                              timestamp_seconds=ts, time_interval_seconds=torch.full(x.shape[:2], 86400, device=dev))
        fc = f.forecast(mt, prediction_length=LH, num_samples=256, samples_per_batch=256)
        out.append(fc.samples[:, 0].permute(0, 2, 1).float().cpu().numpy())
    return np.concatenate(out), "samples"


RUN = {"chronos": run_chronos, "timesfm": run_timesfm, "moirai": run_moirai, "hf": run_hf, "toto": run_toto}


def main():
    q = argparse.ArgumentParser()
    q.add_argument("--family", required=True, choices=list(RUN))
    q.add_argument("--model", required=True)
    q.add_argument("--market", required=True)
    q.add_argument("--tag", required=True)
    q.add_argument("--max_rows", type=int, default=0, help="Smoke tests only; the result is not written to results/")
    q.add_argument("--lh", type=int, default=10, help="Forecast horizon (12 for Time-MMD)")
    a = q.parse_args()
    global LH
    LH = a.lh
    torch.manual_seed(0)
    r = np.load(f"px_rows/{a.market}.npz")
    ctx = r["hist"][:a.max_rows or None, :, 3].astype(np.float32)
    t = time.time()
    px, kind = RUN[a.family](a.model, ctx, "cuda")
    assert px.shape[0] == len(ctx) and px.shape[-1] == LH, px.shape
    l0 = r["l0"][:len(ctx), None, None]
    paths = np.log(np.maximum(px, l0 * 1e-3) / l0).astype(np.float32)
    if a.max_rows:
        print(a.tag, px.shape, kind, "median 10-bar return", float(np.median(paths[:, :, -1])), flush=True)
        return
    os.makedirs("px_tsfm_out", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    np.savez(f"px_tsfm_out/{a.tag}.npz", paths=paths, kind=kind)
    json.dump(dict(args=vars(a), kind=kind, K=int(paths.shape[1]), sec=time.time() - t), open(f"results/{a.tag}.json", "w"))
    print(a.tag, paths.shape, kind, flush=True)


if __name__ == "__main__":
    main()
