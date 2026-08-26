# -*- coding: utf-8 -*-
"""导出纯多头选股的日频超额序列(供画超额曲线).

口径与 backtest.py --side long --style tranche5 完全一致:
  组合 = top decile 等权、5日tranche(每天轮换1/5);
  基准 = 当日成分股**指数权重**加权(指增产品的真实基准);
  超额 = 组合日收益 − 基准日收益;成本按 |Δw| × 单边费率扣减。

输出: results/<index>/curve_long_<pred>.csv  [date, ex_gross, ex_net10, ex_net20, ex_net30, turnover]

用法: /usr/bin/python3 export_curve.py --index hs300 csi500
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
from dsutil import ds_dir, pred_dir
from btutil import NQ, tie_broken_decile
from styleutil import build_styles, neutralize

ANN = 242


def curve(index, pred_name="lgbm_y5d_rank", neutral=False, window="val"):
    preds = pd.read_parquet(os.path.join(pred_dir(index), f"{pred_name}.parquet"))
    preds["date"] = preds["date"].astype(str)
    lab = pd.read_parquet(os.path.join(ds_dir(index), "ds.parquet"),
                          columns=["date", "symbol", "ret_fwd_1d", "tradable"])
    lab["date"] = lab["date"].astype(str)
    df = preds.merge(lab, on=["date", "symbol"], how="inner")
    if window == "test":
        df = df[df["date"] >= C.TEST_START]        # 测试段一次性评估
    else:
        df = df[(df["date"] >= C.VAL_START) & (df["date"] < C.TEST_START)]

    w = C.load_weights([index]).rename(columns={"symbol": "stock_code"})
    w["date"] = w["date"].astype(str)
    w["symbol"] = w["stock_code"].str[:6]
    df = df.merge(w[["date", "symbol", "weight"]], on=["date", "symbol"], how="left")
    dates = sorted(df["date"].unique())
    if window != "test":
        C.assert_dev_dates(dates, "export_curve")
    score_col = "score"
    if neutral:                      # 风格中性口径: 分数对五风格取横截面残差
        st = build_styles(ds_dir(index), dates)
        df = df.merge(st, on=["date", "symbol"], how="left")
        df["score_neu"], _ = neutralize(df)
        score_col = "score_neu"
        df = df.dropna(subset=[score_col])

    prev_w, state = {}, {"subs": [dict() for _ in range(5)]}
    rows = []
    for ti, (d, g) in enumerate(df.groupby("date", sort=True)):
        g = g.dropna(subset=[score_col, "ret_fwd_1d", "weight"])
        if len(g) < 30:
            continue
        syms = g["symbol"].values
        q = tie_broken_decile(g[score_col].values, d)
        tr = g["tradable"].values == 1
        top = set(syms[(q == NQ - 1) & tr])
        if not top:
            continue
        # tranche5: 5个子组合轮换, 每天只重建1个
        k = ti % 5
        state["subs"][k] = {s: 1.0 / len(top) for s in top}
        wp_d = {}
        for sb in state["subs"]:
            for s, wi in sb.items():
                wp_d[s] = wp_d.get(s, 0) + wi / 5.0
        wp = np.array([wp_d.get(s, 0.0) for s in syms])
        if wp.sum() <= 0:
            continue
        wp = wp / wp.sum()

        # 基准 = 指数权重
        wb = g["weight"].values / g["weight"].sum()
        r = np.nan_to_num(g["ret_fwd_1d"].values)

        # 换手: 扣除持仓自然漂移
        cur_prev = np.array([prev_w.get(s, 0.0) for s in syms])
        drift = cur_prev / cur_prev.sum() if cur_prev.sum() > 0 else np.zeros(len(g))
        extra = sum(v for s, v in prev_w.items() if s not in set(syms))
        tw = 0.5 * (np.abs(wp - drift).sum() + extra) * 2

        ex = float(np.sum(wp * r) - np.sum(wb * r))
        rows.append(dict(date=d, ex_gross=ex, turnover=tw,
                         port=float(np.sum(wp * r)), bench=float(np.sum(wb * r))))
        grown = wp * (1 + r)
        prev_w = dict(zip(syms, grown / grown.sum()))

    c = pd.DataFrame(rows)
    for bp in (10, 20, 30):
        c[f"ex_net{bp}"] = c["ex_gross"] - c["turnover"] * bp / 1e4
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", nargs="*", default=["hs300", "csi500"])
    ap.add_argument("--pred", default="lgbm_y5d_rank")
    ap.add_argument("--neutral", action="store_true", help="风格中性口径")
    ap.add_argument("--window", default="val", choices=["val", "test"])
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    for idx in args.index:
        c = curve(idx, args.pred, neutral=args.neutral, window=args.window)
        d = os.path.join(RESULTS, idx)
        os.makedirs(d, exist_ok=True)
        fp = os.path.join(d, f"curve_long_{'neutral' if args.neutral else 'raw'}{args.suffix}.csv")
        c.to_csv(fp, index=False)
        ann = c["ex_net20"].mean() * ANN
        ir = c["ex_net20"].mean() / c["ex_net20"].std() * np.sqrt(ANN)
        te = c["ex_gross"].std() * np.sqrt(ANN)
        print(f"{idx}: {len(c)}天 | 超额毛 {c['ex_gross'].mean()*ANN*100:+.2f}% | "
              f"净20bp {ann*100:+.2f}% | IR {ir:.2f} | TE {te*100:.2f}% | "
              f"换手 {c['turnover'].mean():.3f} → {fp}", flush=True)


if __name__ == "__main__":
    main()
