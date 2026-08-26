# -*- coding: utf-8 -*-
"""上下尾特征归因对比: 预测"谁会跌"和"谁会涨"用的是同一批因子吗?

方法(三条互相印证的证据链, 全部只用 train 段, 不碰 val/test):
  1. 对称重要性 vs 不对称重要性 — 训两个二分类器(下尾 vs 上尾), 比较 gain 榜的秩相关;
     若秩相关高 → 同一批因子既predict跌又predict涨(信号对称), 分开建模无意义;
     若低 → 存在"专属下跌因子", 值得单独建模。
  2. 单因子尾部条件 IC — 对每个特征分别算它在**下尾子样本**和**上尾子样本**里的
     判别力(AUC), 找出下尾判别力显著高于上尾的因子。
  3. 经济含义分组 — 按因子名关键词粗分族(涨跌停/流动性/侵略性/羊群/情绪...),
     看哪一族在下尾侧更重要。

用法: IDXML_DS_ROOT=../../data /usr/bin/python3 tail_attribution.py --threads 60
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
from dsutil import load_ds
from btutil import q_thresh

warnings.filterwarnings("ignore")

TAIL_Q = 0.20            # ⚠️ y5d_rank 是中心化秩 ∈[-0.5,+0.5]
TAIL_T = q_thresh(TAIL_Q)  # = -0.30
FAMILIES = {
    "涨跌停/封板": ["limit", "uplimit", "downlimit", "seal", "board"],
    "流动性/枯竭": ["liquidity", "depth", "resil", "evapor", "thin", "spread"],
    "侵略性/主动": ["aggr", "aggress", "sweep", "penetrat", "impact", "arrival"],
    "撤单/犹豫": ["cancel", "balk", "hesit", "abandon", "withdraw", "retreat"],
    "羊群/同步": ["herd", "swarm", "sync", "conform", "consensus", "crowd", "network"],
    "情绪/行为偏差": ["regret", "disposition", "anchor", "face", "humble", "defer",
                "bias", "overpay", "deindivid", "prospect"],
    "队列/排队": ["queue", "jump", "position", "occupancy", "column"],
    "大单/分单": ["size", "child", "quantiz", "block", "tail"],
}


def family_of(name):
    base = name.split("|")[0].lower()
    for fam, kws in FAMILIES.items():
        if any(k in base for k in kws):
            return fam
    return "其他"


def train_side(X, y_rank, side, threads, seed=0):
    """side='down': 标签=落入最差20%; side='up': 标签=落入最好20%."""
    import lightgbm as lgb
    y = (y_rank <= TAIL_T) if side == "down" else (y_rank >= -TAIL_T)
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                           min_child_samples=200, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.6, reg_lambda=1.0, random_state=seed,
                           n_jobs=threads, verbose=-1)
    m.fit(X, y.astype(np.int8))
    return m.booster_.feature_importance(importance_type="gain")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--index", required=True,

                    help="hs300/csi500/csi1000/csi2000 — 每个指数域独立建模")
    ap.add_argument("--threads", type=int, default=60)
    args = ap.parse_args()
    globals()["args"] = args    # build()/results_dir() 等模块级函数要用

    ds, feats = load_ds(args.index)
    # 只用 train 段(< VAL_START), 归因是诊断不是评估
    tr = ds[ds["date"] < C.VAL_START]
    C.assert_dev_dates(sorted(tr["date"].unique()), "tail_attribution")
    y = tr["y5d_rank"].values.astype(np.float32)
    ok = ~np.isnan(y)
    X = tr[feats].values.astype(np.float32)[ok]
    y = y[ok]
    print(f"train段 {ok.sum()} 行 × {len(feats)} 特征", flush=True)

    imp_dn = train_side(X, y, "down", args.threads)
    print("下尾模型完成", flush=True)
    imp_up = train_side(X, y, "up", args.threads)
    print("上尾模型完成", flush=True)

    df = pd.DataFrame({"feat": feats, "gain_down": imp_dn, "gain_up": imp_up})
    df["share_down"] = df["gain_down"] / df["gain_down"].sum()
    df["share_up"] = df["gain_up"] / df["gain_up"].sum()
    df["下尾偏好"] = df["share_down"] - df["share_up"]
    df["family"] = df["feat"].map(family_of)
    df["agg"] = df["feat"].str.split("|").str[-1]
    rho = df["gain_down"].corr(df["gain_up"], method="spearman")

    out = ["# 上下尾特征归因对比", "",
           f"> 只用 train 段({C.DATA_START}~{C.VAL_START}之前)训练两个二分类器: "
           f"下尾=未来5日收益最差{TAIL_Q*100:.0f}%, 上尾=最好{TAIL_Q*100:.0f}%。"
           "同特征同超参, 差别只在标签方向。**val/test 未参与**。", "",
           "## 1. 总体: 两侧用的是同一批因子吗?", "",
           f"- 上下尾重要性 **Spearman 秩相关 = {rho:.3f}**", ""]
    if rho > 0.8:
        out.append("→ **高度重合**: 同一批因子既预测跌也预测涨, 信号本质是对称的; "
                   "单独训下尾模型的增量主要来自**目标函数聚焦**而非「发现了新因子」。")
    elif rho > 0.5:
        out.append("→ **中等重合**: 主干因子共用, 但存在一批侧重某一端的因子, "
                   "不对称建模有一定空间。")
    else:
        out.append("→ **分化明显**: 存在「专属下跌因子」, 单独建模下尾有结构性理由。")
    out.append("")

    out += ["## 2. 最偏向下尾的 15 个特征(share_down − share_up)", "",
            "| 特征 | 族 | 下尾份额 | 上尾份额 | 偏好差 |", "|---|---|---|---|---|"]
    for _, r in df.nlargest(15, "下尾偏好").iterrows():
        out.append(f"| `{r.feat}` | {r.family} | {r.share_down*100:.2f}% | "
                   f"{r.share_up*100:.2f}% | **{r['下尾偏好']*100:+.2f}pp** |")

    out += ["", "## 3. 最偏向上尾的 10 个特征(对照)", "",
            "| 特征 | 族 | 下尾份额 | 上尾份额 | 偏好差 |", "|---|---|---|---|---|"]
    for _, r in df.nsmallest(10, "下尾偏好").iterrows():
        out.append(f"| `{r.feat}` | {r.family} | {r.share_down*100:.2f}% | "
                   f"{r.share_up*100:.2f}% | {r['下尾偏好']*100:+.2f}pp |")

    fam = df.groupby("family")[["share_down", "share_up"]].sum()
    fam["偏好差"] = fam["share_down"] - fam["share_up"]
    fam = fam.sort_values("偏好差", ascending=False)
    out += ["", "## 4. 按经济含义分族", "",
            "| 因子族 | 下尾份额 | 上尾份额 | 偏好差 |", "|---|---|---|---|"]
    for f_, r in fam.iterrows():
        out.append(f"| {f_} | {r.share_down*100:.1f}% | {r.share_up*100:.1f}% | "
                   f"**{r['偏好差']*100:+.1f}pp** |")

    agg = df.groupby("agg")[["share_down", "share_up"]].sum()
    agg["偏好差"] = agg["share_down"] - agg["share_up"]
    out += ["", "## 5. 按日内聚合方式", "",
            "| 聚合 | 下尾份额 | 上尾份额 | 偏好差 |", "|---|---|---|---|"]
    for a, r in agg.sort_values("偏好差", ascending=False).iterrows():
        out.append(f"| {a} | {r.share_down*100:.1f}% | {r.share_up*100:.1f}% | "
                   f"{r['偏好差']*100:+.1f}pp |")
    out.append("")

    df.sort_values("下尾偏好", ascending=False).to_csv(
        os.path.join(results_dir(args.index), "上下尾归因_全特征.csv"), index=False)
    txt = "\n".join(out) + "\n"
    with open(os.path.join(results_dir(args.index), "上下尾特征归因对比.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
