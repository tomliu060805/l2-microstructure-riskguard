# -*- coding: utf-8 -*-
"""特征重要性稳定性与归因分析.

输出:
  1) 跨 refit 块的重要性稳定性(相邻块 Spearman 秩相关; 全期 top-K 重合率)
  2) 按聚合类型(mean/std/lh)的重要性份额
  3) 因子级(合并3个聚合)top20 榜, 附 train段单因子IC
结果写 results/importance_report.md
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
RESULTS = os.path.join(HERE, "..", "results")
import common as C

IMP = os.path.join(C.CACHE, "preds", "importance_lgbm_y5d_rank_imp.parquet")


def main():
    imp = pd.read_parquet(IMP)
    imp = imp.reindex(sorted(imp.columns), axis=1)
    blocks = list(imp.columns)

    # 1) 稳定性
    adj = [imp[blocks[i]].corr(imp[blocks[i + 1]], method="spearman")
           for i in range(len(blocks) - 1)]
    first_last = imp[blocks[0]].corr(imp[blocks[-1]], method="spearman")
    K = 50
    tops = [set(imp[b].nlargest(K).index) for b in blocks]
    overlap_adj = [len(tops[i] & tops[i + 1]) / K for i in range(len(tops) - 1)]
    persist = pd.Series(np.concatenate([list(t) for t in tops])).value_counts()
    always = (persist == len(blocks)).sum()

    # 2) 聚合类型份额
    agg_type = pd.Series({f: f.split("|")[-1] if "|" in f else "other" for f in imp.index})
    mean_imp = imp.mean(1)
    share = mean_imp.groupby(agg_type).sum() / mean_imp.sum()
    cnt = agg_type.value_counts()

    # 3) 因子级榜
    base = pd.Series({f: f.split("|")[0] for f in imp.index})
    fac_imp = mean_imp.groupby(base).sum().sort_values(ascending=False)

    lines = ["# LGBM 特征重要性稳定性与归因 (gain, 3种子均值, 25个refit块)", ""]
    lines += ["## 1. 时间稳定性", "",
              f"- 相邻块重要性 Spearman 秩相关: 均值 **{np.mean(adj):.3f}** "
              f"(范围 {np.min(adj):.3f}~{np.max(adj):.3f})",
              f"- 首块 vs 末块(2018 vs 2024): **{first_last:.3f}**",
              f"- top{K} 相邻块重合率: 均值 **{np.mean(overlap_adj)*100:.0f}%**",
              f"- 全部 {len(blocks)} 个块都进 top{K} 的特征: **{always} 个**",
              "",
              "解读: 秩相关高=模型学到的结构跨市场状态稳定(而非每次refit都换一批特征), "
              "这是信号可信的必要条件; 若接近0则说明纯拟合噪声。", ""]
    lines += ["## 2. 日内聚合类型的重要性份额", "",
              "| 聚合 | 含义 | 特征数 | 重要性份额 |", "|---|---|---|---|"]
    meaning = {"mean": "全日均值", "std": "日内波动", "lh": "尾盘14:00-14:55均值", "other": "coverage"}
    for k in ["mean", "std", "lh", "other"]:
        if k in share.index:
            lines.append(f"| {k} | {meaning[k]} | {cnt.get(k,0)} | {share[k]*100:.1f}% |")
    lines += ["", "## 3. 因子级 top20 (三个聚合的gain求和)", "",
              "| # | 因子 | 重要性份额 |", "|---|---|---|"]
    tot = fac_imp.sum()
    for i, (f, v) in enumerate(fac_imp.head(20).items(), 1):
        lines.append(f"| {i} | `{f}` | {v/tot*100:.2f}% |")
    lines += ["",
              f"- 头部集中度: top20因子占 **{fac_imp.head(20).sum()/tot*100:.1f}%**, "
              f"top50占 **{fac_imp.head(50).sum()/tot*100:.1f}%** (共{len(fac_imp)}个因子)", ""]

    out = os.path.join(RESULTS, "importance_report.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
