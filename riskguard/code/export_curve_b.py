# -*- coding: utf-8 -*-
"""导出规避策略(项目B)的日频超额序列 — 原始与风格中性两种口径.

口径: 持有指数全部成分、剔除预测最差 K%(不可交易时维持基准权重),
      权重按剩余成分再归一化; 基准 = 同域指数权重加权;
      换手扣除基准权重的自然漂移; 成本按 |Δw| × 单边费率。

输出: results/<index>/curve_avoid_<mode>.csv  [date, ex_gross, ex_net10/20/30, turnover]

用法: /usr/bin/python3 export_curve_b.py --index csi1000 gz2000 --frac 0.10
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
RESULTS = os.path.join(HERE, "..", "results")
import common as C
from dsutil import ds_dir
from styleutil import build_styles, neutralize
from short_side_exclude import build

ANN = 242


def daily_curve(df, frac, score_col):
    """剔除底部 frac(仅可交易时), 返回逐日超额与换手."""
    prev_w, rows = {}, []
    for d, g in df.groupby("date", sort=True):
        g = g.dropna(subset=["weight", "ret_fwd_1d", score_col])
        if len(g) < 50:
            continue
        syms = g["symbol"].values
        wb = g["weight"].values / g["weight"].sum()
        pct = g[score_col].rank(pct=True).values
        can = g["tradable"].values == 1
        wp = wb * (~((pct <= frac) & can))
        if wp.sum() <= 0:
            continue
        wp = wp / wp.sum()
        cur_prev = np.array([prev_w.get(s, 0.0) for s in syms])
        drift = cur_prev / cur_prev.sum() if cur_prev.sum() > 0 else np.zeros(len(g))
        extra = sum(v for s, v in prev_w.items() if s not in set(syms))
        tw = 0.5 * (np.abs(wp - drift).sum() + extra) * 2
        r = np.nan_to_num(g["ret_fwd_1d"].values)
        rows.append(dict(date=d, ex_gross=float(np.sum(wp * r) - np.sum(wb * r)), turnover=tw))
        grown = wp * (1 + r)
        prev_w = dict(zip(syms, grown / grown.sum()))
    c = pd.DataFrame(rows)
    for bp in (10, 20, 30):
        c[f"ex_net{bp}"] = c["ex_gross"] - c["turnover"] * bp / 1e4
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", nargs="*", default=C.RISK_INDEXES)
    ap.add_argument("--pred", default="lgbm_y5d_rank")
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--window", default="val", choices=["val", "test"])
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    for idx in args.index:
        df, dates = build(idx, pred_name=args.pred, window=args.window)
        st = build_styles(ds_dir(idx), dates)
        df = df.merge(st, on=["date", "symbol"], how="left")
        df["score_neu"], _ = neutralize(df)
        outdir = os.path.join(RESULTS, idx)
        os.makedirs(outdir, exist_ok=True)
        for mode, col in [("raw", "score"), ("neutral", "score_neu")]:
            d2 = df.sort_values(["symbol", "date"]).copy()
            d2["score_sm"] = d2.groupby("symbol")[col].transform(
                lambda s: s.rolling(5, min_periods=1).mean())
            d2 = d2.sort_values(["date", "symbol"]).reset_index(drop=True)
            c = daily_curve(d2, args.frac, "score_sm")
            fp = os.path.join(outdir, f"curve_avoid_{mode}{args.suffix}.csv")
            c.to_csv(fp, index=False)
            ann = c["ex_net20"].mean() * ANN
            ir = c["ex_net20"].mean() / c["ex_net20"].std() * np.sqrt(ANN)
            print(f"{idx}/{mode}: 净20bp {ann*100:+.2f}% | IR {ir:.2f} | "
                  f"TE {c['ex_gross'].std()*np.sqrt(ANN)*100:.2f}% | 换手 {c['turnover'].mean():.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
