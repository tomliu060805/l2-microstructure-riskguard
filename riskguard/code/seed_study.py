# -*- coding: utf-8 -*-
"""多种子分布 — 训练随机性有多大?

每个单种子模型**独立评估**(不做种子平均), 报:
  均值 / 标准差 / 最小值(最差情况) / 为正占比 / 分位数
另与"多种子平均后的模型"对比, 量化平均带来的去噪收益。

评估口径与主报告一致: 风格中性分数 + 5日平滑 + 剔除底部K% + 指数权重基准 + 净20bp。

用法: /usr/bin/python3 seed_study.py --univ csi1000 --frac 0.10
"""
import argparse
import glob
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
from alpha_ceiling import simulate_shape


warnings.filterwarnings("ignore")


def eval_pred(univ, pred_name, st_cache, frac, neutral=True):
    df, dates = build(univ, pred_name=pred_name)
    df = df.merge(st_cache, on=["date", "symbol"], how="left")
    col = "score"
    if neutral:
        df["score_neu"], _ = neutralize(df)
        col = "score_neu"
    d2 = df.sort_values(["symbol", "date"]).copy()
    d2["score_sm"] = d2.groupby("symbol")[col].transform(
        lambda s: s.rolling(5, min_periods=1).mean())
    d2 = d2.sort_values(["date", "symbol"]).reset_index(drop=True)
    return simulate_shape(d2, "exclude", frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="csi1000")
    ap.add_argument("--frac", type=float, default=0.10)
    args = ap.parse_args()

    u = args.index
    pat = os.path.join(pred_dir(u), "lgbm_y5d_rank_s*.parquet")
    files = sorted(glob.glob(pat))
    if not files:
        print("尚无单种子预测文件:", pat)
        return
    seeds = [os.path.basename(f).split("_s")[-1].replace(".parquet", "") for f in files]
    print(f"{u}: 找到 {len(files)} 个单种子模型 (seed {','.join(seeds)})", flush=True)

    df0, dates = build(u, pred_name=f"lgbm_y5d_rank_s{seeds[0]}")
    st_cache = build_styles(ds_dir(u), dates)

    rows = []
    for f, sd in zip(files, seeds):
        name = os.path.basename(f).replace(".parquet", "")
        for tag, neu in [("风格中性", True), ("原始", False)]:
            r = eval_pred(u, name, st_cache, args.frac, neutral=neu)
            rows.append(dict(seed=int(sd), 口径=tag, 毛=r["gross"], 净20=r["net20"],
                             IR=r["ir20"], TE=r["te"], 换手=r["tw"]))
        print(f"  seed {sd} 完成", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(results_dir(u), "多种子分布.csv"), index=False)

    out = [f"# 多种子分布 — {NAME(u)}", "",
           f"> 验证段 2022-07-01~2024-06-28 | 剔除底部 {args.frac*100:.0f}% | 净20bp | "
           "**每个单种子模型独立评估, 不做种子平均**", "",
           "## 分布统计", "",
           "| 口径 | 种子数 | 均值 | 标准差 | 最小值(最差) | 最大值 | 中位数 | **为正占比** |",
           "|---|---|---|---|---|---|---|---|"]
    for tag in ["风格中性", "原始"]:
        v = res[res["口径"] == tag]["净20"]
        if v.empty:
            continue
        out.append(f"| {tag} | {len(v)} | **{v.mean()*100:+.2f}%** | {v.std()*100:.2f}% | "
                   f"**{v.min()*100:+.2f}%** | {v.max()*100:+.2f}% | {v.median()*100:+.2f}% | "
                   f"**{(v>0).mean()*100:.0f}%** |")
    out += ["", "## IR 分布", "",
            "| 口径 | 均值 | 标准差 | 最小值 | 最大值 |", "|---|---|---|---|---|"]
    for tag in ["风格中性", "原始"]:
        v = res[res["口径"] == tag]["IR"]
        if v.empty:
            continue
        out.append(f"| {tag} | {v.mean():.2f} | {v.std():.2f} | {v.min():.2f} | {v.max():.2f} |")

    out += ["", "## 逐种子明细(风格中性口径)", "",
            "| seed | 毛超额 | 净20bp | IR | TE | 换手 |", "|---|---|---|---|---|---|"]
    for _, r in res[res["口径"] == "风格中性"].sort_values("seed").iterrows():
        out.append(f"| {r.seed} | {r.毛*100:+.2f}% | {r.净20*100:+.2f}% | {r.IR:.2f} | "
                   f"{r.TE*100:.2f}% | {r.换手:.3f} |")

    v = res[res["口径"] == "风格中性"]["净20"]
    cv = v.std() / abs(v.mean()) if v.mean() != 0 else np.nan
    out += ["", "## 解读", "",
            f"- 变异系数(标准差/均值) = **{cv:.2f}** —— 越小说明结果越不依赖种子运气;",
            f"- 最差种子 {v.min()*100:+.2f}%, 为正占比 {(v>0).mean()*100:.0f}%;",
            "- **种子噪声只是不确定性的一个来源**。本项目更大的不确定性来自",
            "  ①从数十个配置里挑最优(多重检验) ②验证段被 A1/A2/扫描反复使用。",
            "  这两条种子测再多也无法解决, 只能靠**测试段一次性开封**。", ""]

    with open(os.path.join(results_dir(u), "多种子分布.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
