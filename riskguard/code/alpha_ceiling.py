# -*- coding: utf-8 -*-
"""小盘域 alpha 上限探索 — 风格中性口径下能做到多少超额?

## 背景
A1 证明原口径的超额 4/5 是低波动风格暴露, 但**残余 alpha 随市值下降增强**
(留存率 19%→31%→46%)。本脚本回答: **在风格中性口径下, 中证1000/2000 的
选股超额上限是多少?**

## 三个维度扫描
  信号   对称模型 vs 下尾模型 × 原始分数 vs 风格中性分数
  形态   剔除底部K%(纯规避) / 剔除+超配顶部(双边倾斜) / 纯多头top-N
  参数   K ∈ {10,20,30,40}%, 倾斜强度

## 关键口径
- 超额相对**各自指数权重基准**;
- 成本用 A2 实测校准值(单边 13bp ≈ 5亿规模, 另报 20/25bp);
- 所有配置共享同一 5 日平滑与同一回测器, 唯一变量是信号与组合形态。

用法: /usr/bin/python3 alpha_ceiling.py [--index csi1000 gz2000]
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
from short_side_exclude import build, ANN
from dsutil import ds_dir, pred_dir
from style_neutralize import build_styles, neutralize, NAME

warnings.filterwarnings("ignore")

PREDS = {
    "csi500":  {"对称": "lgbm_y5d_rank",         "下尾": "tail_clf_w_y5d_csi500"},
    "csi1000": {"对称": "lgbm_y5d_rank_csi1000", "下尾": "tail_clf_w_y5d_csi1000"},
    "csi2000": {"对称": "lgbm_y5d_rank_csi2000", "下尾": "tail_clf_w_y5d_csi2000"},
}
PREDS = {"对称": "lgbm_y5d_rank", "下尾": "tail_clf_w_y5d"}
COSTS = (13, 20, 25)     # A2 实测: 5亿≈13bp, 20亿≈18bp, 50亿≈25bp


def simulate_shape(df, shape, frac, tilt=1.0, score_col="score_sm"):
    """组合形态:
      exclude  剔除底部frac, 其余按基准权重(纯规避, 现有主口径)
      tilt     剔除底部frac + 顶部frac超配tilt倍(双边倾斜)
      topn     只持有顶部frac(纯多头集中, TE最大)
    """
    prev_w, ex_r, tw_l = {}, [], []
    for d, g in df.groupby("date", sort=True):
        g = g.dropna(subset=["weight", "ret_fwd_1d", score_col])
        if len(g) < 50:
            continue
        syms = g["symbol"].values
        wb = g["weight"].values / g["weight"].sum()
        pct = g[score_col].rank(pct=True).values
        can = g["tradable"].values == 1

        if shape == "exclude":
            keep = ~((pct <= frac) & can)
            wp = wb * keep
        elif shape == "tilt":
            keep = ~((pct <= frac) & can)
            mult = np.ones(len(g))
            mult[(pct >= 1 - frac) & can] = tilt
            wp = wb * keep * mult
        elif shape == "topn":
            sel = (pct >= 1 - frac) & can
            wp = wb * sel
        else:
            raise ValueError(shape)
        if wp.sum() <= 0:
            continue
        wp = wp / wp.sum()

        cur_prev = np.array([prev_w.get(s, 0.0) for s in syms])
        drift = cur_prev / cur_prev.sum() if cur_prev.sum() > 0 else np.zeros(len(g))
        extra = sum(v for s, v in prev_w.items() if s not in set(syms))
        tw_l.append(0.5 * (np.abs(wp - drift).sum() + extra) * 2)

        r = np.nan_to_num(g["ret_fwd_1d"].values)
        ex_r.append(float(np.sum(wp * r) - np.sum(wb * r)))
        grown = wp * (1 + r)
        prev_w = dict(zip(syms, grown / grown.sum()))

    ex_r, tw = np.array(ex_r), np.array(tw_l)
    te = ex_r.std() * np.sqrt(ANN)
    out = {"gross": ex_r.mean() * ANN, "te": te, "tw": tw.mean(), "n": len(ex_r)}
    for bp in COSTS:
        v = ex_r - tw * bp / 1e4
        out[f"net{bp}"] = v.mean() * ANN
        out[f"ir{bp}"] = v.mean() / (v.std() + 1e-12) * np.sqrt(ANN)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", nargs="*", default=C.RISK_INDEXES)
    ap.add_argument("--signals", nargs="*", default=None, help="只跑指定信号(并行切分用)")
    ap.add_argument("--fine", action="store_true", help="细网格+更强倾斜")
    ap.add_argument("--out", default="小盘域alpha上限", help="输出文件名(不含扩展名)")
    args = ap.parse_args()

    rows = []
    for u in args.index:
        if not os.path.exists(os.path.join(ds_dir(u), "ds.parquet")):
            continue
        print(f"=== {u}", flush=True)
        st_cache = None
        for kind, pred in PREDS.items():
            if args.signals and kind not in args.signals:
                continue
            if not os.path.exists(os.path.join(pred_dir(u), f"{pred}.parquet")):
                continue
            df, dates = build(u, pred_name=pred)
            if st_cache is None:
                st_cache = build_styles(ds_dir(u), dates)
            df = df.merge(st_cache, on=["date", "symbol"], how="left")
            df["score_neu"], _ = neutralize(df)

            for neu, col in [("原始", "score"), ("中性化", "score_neu")]:
                d2 = df.sort_values(["symbol", "date"]).copy()
                d2["score_sm"] = d2.groupby("symbol")[col].transform(
                    lambda s: s.rolling(5, min_periods=1).mean())
                d2 = d2.sort_values(["date", "symbol"]).reset_index(drop=True)
                grids = ([("exclude", (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50), (1.0,)),
                          ("tilt", (0.10, 0.20, 0.30, 0.40), (1.5, 2.0, 3.0, 5.0)),
                          ("topn", (0.20, 0.30, 0.40, 0.50, 0.70), (1.0,))]
                         if args.fine else
                         [("exclude", (0.10, 0.20, 0.30, 0.40), (1.0,)),
                          ("tilt", (0.20, 0.30), (1.5, 2.0)),
                          ("topn", (0.30, 0.50), (1.0,))])
                for shape, fracs, tilts in grids:
                    for fr in fracs:
                        for tl in tilts:
                            r = simulate_shape(d2, shape, fr, tl)
                            rows.append(dict(域=u, 信号=kind, 口径=neu, 形态=shape,
                                             K=fr, tilt=tl, **r))
                print(f"  {kind}/{neu} 完成", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, f"{args.out}_扫描.csv"), index=False)

    out = ["# 小盘域 alpha 上限探索(风格中性口径)", "",
           "> 验证段 2022-07-01~2024-06-28 | 超额相对各自指数权重基准 | "
           "成本按 A2 实测校准(13bp≈5亿规模 / 20bp≈20亿 / 25bp≈50亿) | **测试段未触碰**", "",
           "**核心问题**: A1 证明原口径超额 4/5 是低波动暴露, 那么在**严格风格中性**口径下, "
           "小盘域的选股超额上限是多少?", ""]

    for u in args.index:
        sub = res[res["域"] == u]
        if sub.empty:
            continue
        out += [f"## {NAME(u)}", "",
                "### 风格中性口径(真实选股 alpha)", "",
                "| 信号 | 形态 | K | 毛超额 | 净13bp | **净20bp** | IR(净20) | TE | 换手 |",
                "|---|---|---|---|---|---|---|---|---|"]
        s2 = sub[sub["口径"] == "中性化"].sort_values("net20", ascending=False)
        for _, r in s2.head(10).iterrows():
            sh = {"exclude": "剔除底部", "tilt": f"剔底+超配{r.tilt:g}×", "topn": "纯多头top"}[r.形态]
            out.append(f"| {r.信号} | {sh} | {r.K*100:.0f}% | {r.gross*100:+.2f}% | "
                       f"{r.net13*100:+.2f}% | **{r.net20*100:+.2f}%** | **{r.ir20:.2f}** | "
                       f"{r.te*100:.2f}% | {r.tw:.3f} |")
        best = s2.iloc[0]
        out += ["", f"**最优(中性化口径)**: {best.信号}模型 + "
                f"{ {'exclude':'剔除底部','tilt':'剔底+超配','topn':'纯多头top'}[best.形态] }"
                f"{best.K*100:.0f}% → 净20bp **{best.net20*100:+.2f}%**, IR **{best.ir20:.2f}**, "
                f"TE {best.te*100:.2f}%", ""]

        out += ["### 原始口径(含风格暴露, 对照)", "",
                "| 信号 | 形态 | K | 毛超额 | **净20bp** | IR | TE |",
                "|---|---|---|---|---|---|---|"]
        s1 = sub[sub["口径"] == "原始"].sort_values("net20", ascending=False)
        for _, r in s1.head(5).iterrows():
            sh = {"exclude": "剔除底部", "tilt": f"剔底+超配{r.tilt:g}×", "topn": "纯多头top"}[r.形态]
            out.append(f"| {r.信号} | {sh} | {r.K*100:.0f}% | {r.gross*100:+.2f}% | "
                       f"**{r.net20*100:+.2f}%** | {r.ir20:.2f} | {r.te*100:.2f}% |")
        out.append("")

    with open(os.path.join(RESULTS, f"{args.out}.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
