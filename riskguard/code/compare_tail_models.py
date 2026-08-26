# -*- coding: utf-8 -*-
"""下尾专用模型 vs 对称模型: 在"剔除底部"这个落地形态下对比.

评估口径统一为项目2 的最优配置(5日信号平滑 + 剔除底部K% + 指数权重基准),
把每个模型的分数插进同一个回测器, 差别只在信号来源 → 增量可归因于目标函数。

另报下尾专属指标:
  - 下尾捕获率 Recall@K — 真实最差20%的票, 有多少被模型排进了预测最差K%
  - 底部组的平均实际收益 — 预测最差K%的票, 实际跌了多少

用法: IDXML_DS_ROOT=../../data /usr/bin/python3 compare_tail_models.py
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
RESULTS = os.path.join(HERE, "..", "results")


def results_dir(index):
    d = os.path.join(RESULTS, index)
    os.makedirs(d, exist_ok=True)
    return d
import common as C
from dsutil import ds_dir, pred_dir
import short_side_exclude as SSE
from short_side_exclude import build, simulate, ANN

warnings.filterwarnings("ignore")

MODELS = [
    ("lgbm_y5d_rank", "对称回归(项目1主模型)"),
    ("tail_clf_y5d", "下尾二分类"),
    ("tail_clf_w_y5d", "下尾二分类+深跌加权"),
    ("tail_quant_y5d", "分位数回归(q=0.10)"),
]


def tail_stats(df, score_col, k=0.20, true_q=0.20):
    """下尾捕获率与底部组实际收益(用未来5日收益秩作真值)."""
    rec, botret = [], []
    for d, g in df.groupby("date", sort=True):
        g = g.dropna(subset=[score_col, "y5d_rank_true", "ret_fwd_1d"])
        if len(g) < 50:
            continue
        n = len(g)
        pred_bot = g[score_col].rank(pct=True) <= k
        # y5d_rank_true 是中心化秩(值域[-0.5,+0.5]), 不能直接与百分位阈值比较,
        # 否则 <=0.20 会选中约 70% 的票, 捕获率被稀释成伪随机的 20%。统一转百分位。
        true_bot = g["y5d_rank_true"].rank(pct=True) <= true_q
        if true_bot.sum() > 0:
            rec.append(float((pred_bot & true_bot).sum() / true_bot.sum()))
        botret.append(float(g.loc[pred_bot, "ret_fwd_1d"].mean()))
    return float(np.mean(rec)), float(np.mean(botret) * ANN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True,
                    help="hs300/csi500/csi1000/gz2000")
    args = ap.parse_args()
    globals()["args"] = args    # build() 等模块级函数也要用
    SSE.args = args             # 被导入模块 short_side_exclude 的 build()/simulate() 同样需要

    base_df, dates = build()          # 含 score(项目A模型) + 平滑分数
    truth = pd.read_parquet(os.path.join(ds_dir(args.index), "ds.parquet"),
                            columns=["date", "symbol", "y5d_rank"])
    truth["date"] = truth["date"].astype(str)
    truth = truth.rename(columns={"y5d_rank": "y5d_rank_true"})

    rows = []
    for name, label in MODELS:
        fp = os.path.join(pred_dir(args.index), f"{name}.parquet")
        if not os.path.exists(fp):
            print("跳过(未训练):", name, flush=True)
            continue
        p = pd.read_parquet(fp)
        p["date"] = p["date"].astype(str)
        df = base_df.drop(columns=["score", "score_sm"]).merge(
            p, on=["date", "symbol"], how="inner")
        df = df.sort_values(["symbol", "date"])
        df["score_sm"] = (df.groupby("symbol")["score"]
                          .transform(lambda s: s.rolling(5, min_periods=1).mean()))
        df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
        df = df.merge(truth, on=["date", "symbol"], how="left")

        recall, botret = tail_stats(df, "score_sm")
        for frac in (0.10, 0.30):
            res, te, tw, nd = simulate(df, frac, "smooth")
            r20 = res[20]
            rows.append(dict(模型=label, 剔除比例=f"{frac*100:.0f}%",
                             超额毛=res[0]["ann"], 净20bp=r20["ann"], IR=r20["ir"],
                             TE=te, 换手=tw, 胜率=r20["wr"], 盈利因子=r20["pf"],
                             下尾捕获率=recall, 底部组实际年化=botret, 交易日=nd))
        print("done", name, flush=True)

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(results_dir(args.index), "下尾模型对照.csv"), index=False)

    out = ["# 下尾专用模型 vs 对称模型", "",
           f"> 验证段 {dates[0]}~{dates[-1]}, {len(dates)}天 | 统一口径: 5日信号平滑 + "
           f"剔除底部K% + {C.INDEXES[args.index]['name']}指数权重基准 | **测试段未触碰**", "",
           "所有模型共享完全相同的特征集(741维)、walk-forward调度(63天refit/5天embargo)、"
           "回测框架, **差别只在目标函数** → 增量可归因于「不对称建模」本身。", "",
           "| 模型 | 剔除比例 | 超额毛 | 超额净20bp | IR | TE | 换手 | 胜率 | 盈利因子 | 下尾捕获率 | 底部组实际年化 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res_df.iterrows():
        out.append(f"| {r.模型} | {r.剔除比例} | {r.超额毛*100:+.2f}% | "
                   f"**{r.净20bp*100:+.2f}%** | **{r.IR:.2f}** | {r.TE*100:.2f}% | "
                   f"{r.换手:.3f} | {r.胜率*100:.0f}% | {r.盈利因子:.2f} | "
                   f"{r.下尾捕获率*100:.1f}% | {r.底部组实际年化*100:+.1f}% |")
    out += ["", "**下尾捕获率** = 真实未来5日收益最差20%的票中, 被模型排进预测最差20%的比例",
            "(随机猜为20%);**底部组实际年化** = 被预测为最差20%的票的实际次日收益年化。", ""]

    txt = "\n".join(out) + "\n"
    with open(os.path.join(results_dir(args.index), "下尾模型对照.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
