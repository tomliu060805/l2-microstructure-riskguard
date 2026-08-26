# -*- coding: utf-8 -*-
"""稳健性检验 — 这个超额是真的吗?四个互相独立的角度.

面试常问"你怎么知道不是运气", 单看点估计答不了。本脚本给出:

  ① 时间自助法(moving block bootstrap) — 超额的置信区间与"为正"的概率
     日超额有自相关(5日平滑+持仓延续), 必须用**块**自助而非独立重采样, 否则CI虚窄。
  ② 分年稳定性 — 是不是靠某一年撑起来的
  ③ 配置分布 — 报**全部配置的分布**而不只是最优值, 量化"挑最好的"带来的膨胀
  ④ 多种子分布 — 由 seed_study.py 单独出(需重训, 见该脚本)

用法: /usr/bin/python3 robustness.py [--index csi1000 gz2000]
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
from alpha_ceiling import simulate_shape, PREDS

warnings.filterwarnings("ignore")

# 各域待检验的配置: (信号, 形态, K) —— 取风格中性口径下的代表配置
CONFIGS = {          # 各域待检验代表配置(依据 阶段结果汇总: 小盘窄口径缓冲带最优)
    "hs300":   [("对称", 0.30), ("对称", 0.15)],
    "csi500":  [("对称", 0.30), ("对称", 0.15)],
    "csi1000": [("对称", 0.05), ("对称", 0.30)],
    "gz2000":  [("对称", 0.05), ("对称", 0.30)],
}
BLOCK = 21          # 自助块长(约一个月, 覆盖5日平滑与持仓延续的自相关)
NBOOT = 2000


def daily_excess(df, shape, frac, cost_bp=20):
    """复用 simulate_shape 的逻辑但返回逐日净超额序列."""
    prev_w, ex_r, tw_l, ds = {}, [], [], []
    for d, g in df.groupby("date", sort=True):
        g = g.dropna(subset=["weight", "ret_fwd_1d", "score_sm"])
        if len(g) < 50:
            continue
        syms = g["symbol"].values
        wb = g["weight"].values / g["weight"].sum()
        pct = g["score_sm"].rank(pct=True).values
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
        ex_r.append(float(np.sum(wp * r) - np.sum(wb * r)) - tw * cost_bp / 1e4)
        ds.append(d)
        grown = wp * (1 + r)
        prev_w = dict(zip(syms, grown / grown.sum()))
    return pd.Series(ex_r, index=pd.to_datetime(ds))


def block_bootstrap(x, block=BLOCK, n=NBOOT, seed=0):
    """移动块自助: 保留自相关结构."""
    rng = np.random.RandomState(seed)
    v = np.asarray(x, float)
    v = v[~np.isnan(v)]
    T = len(v)
    nb = int(np.ceil(T / block))
    starts_pool = np.arange(0, T - block + 1)
    means, irs = [], []
    for _ in range(n):
        st = rng.choice(starts_pool, nb, replace=True)
        samp = np.concatenate([v[s:s + block] for s in st])[:T]
        means.append(samp.mean() * ANN)
        irs.append(samp.mean() / (samp.std() + 1e-12) * np.sqrt(ANN))
    return np.array(means), np.array(irs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", nargs="*", default=C.RISK_INDEXES)
    args = ap.parse_args()

    out = ["# 稳健性检验 — 这个超额是运气吗?", "",
           "> 验证段 2022-07-01~2024-06-28(484个交易日) | 风格中性口径 | 成本净20bp",
           "> | **测试段未触碰**", "",
           "面试常问「你怎么知道不是运气」。单看点估计答不了,本报告给出四个独立角度。", ""]

    boot_rows, year_rows = [], []
    for u in args.index:
        if not os.path.exists(os.path.join(ds_dir(u), "ds.parquet")):
            continue
        st_cache = None
        for sig, frac in CONFIGS.get(u, []):
            pred = PREDS[sig]
            if not os.path.exists(os.path.join(pred_dir(u), f"{pred}.parquet")):
                continue
            df, dates = build(u, pred_name=pred)
            if st_cache is None:
                st_cache = build_styles(ds_dir(u), dates)
            df = df.merge(st_cache, on=["date", "symbol"], how="left")
            df["score_neu"], _ = neutralize(df)
            d2 = df.sort_values(["symbol", "date"]).copy()
            d2["score_sm"] = d2.groupby("symbol")["score_neu"].transform(
                lambda s: s.rolling(5, min_periods=1).mean())
            d2 = d2.sort_values(["date", "symbol"]).reset_index(drop=True)

            ex = daily_excess(d2, "exclude", frac)
            m, ir = block_bootstrap(ex)
            name = f"{NAME(u)} {sig}/剔除{frac*100:.0f}%"
            boot_rows.append(dict(配置=name, 点估计=ex.mean() * ANN,
                                  均值=m.mean(), 标准差=m.std(),
                                  p5=np.percentile(m, 5), p95=np.percentile(m, 95),
                                  为正概率=float((m > 0).mean()),
                                  IR点估计=ex.mean() / ex.std() * np.sqrt(ANN),
                                  IR_p5=np.percentile(ir, 5)))
            yr = ex.groupby(ex.index.year).agg(["mean", "count"])
            for y, r in yr.iterrows():
                year_rows.append(dict(配置=name, 年份=y, 年化=r["mean"] * ANN, 天数=int(r["count"])))
            print(f"{name}: 点估计 {ex.mean()*ANN*100:+.2f}% | "
                  f"自助 [{np.percentile(m,5)*100:+.2f}%, {np.percentile(m,95)*100:+.2f}%] | "
                  f"为正 {float((m>0).mean())*100:.0f}%", flush=True)

    bt = pd.DataFrame(boot_rows)
    out += ["## ① 时间自助法(移动块自助, 块长21日, 2000次)", "",
            "日超额有自相关(5日平滑 + 持仓延续),用**块**自助而非独立重采样,否则置信区间虚窄。", "",
            "| 配置 | 点估计 | 自助均值 | 标准差 | 5%分位 | 95%分位 | **为正概率** | IR点估计 | IR的5%分位 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for _, r in bt.iterrows():
        out.append(f"| {r.配置} | {r.点估计*100:+.2f}% | {r.均值*100:+.2f}% | {r.标准差*100:.2f}% | "
                   f"{r.p5*100:+.2f}% | {r.p95*100:+.2f}% | **{r.为正概率*100:.0f}%** | "
                   f"{r.IR点估计:.2f} | {r.IR_p5:.2f} |")

    yr = pd.DataFrame(year_rows)
    out += ["", "## ② 分年稳定性", "", "| 配置 | " +
            " | ".join(str(y) for y in sorted(yr["年份"].unique())) + " |",
            "|---|" + "---|" * yr["年份"].nunique()]
    for cfg, g in yr.groupby("配置", sort=False):
        cells = []
        for y in sorted(yr["年份"].unique()):
            row = g[g["年份"] == y]
            cells.append(f"{row['年化'].iloc[0]*100:+.1f}%" if len(row) else "—")
        out.append(f"| {cfg} | " + " | ".join(cells) + " |")
    out += ["", "(2022 与 2024 为部分年度,天数少,波动大)", ""]

    # ③ 配置分布
    out += ["## ③ 配置分布(量化「挑最好的」带来的膨胀)", "",
            "| 域 | 配置数 | 最优 | 中位数 | 25%分位 | 最差 | **为正占比** |",
            "|---|---|---|---|---|---|---|"]
    for f in sorted(glob.glob(os.path.join(RESULTS, "*alpha上限*_扫描.csv"))):
        d = pd.read_csv(f)
        d = d[d["口径"] == "中性化"]
        if d.empty:
            continue
        for u_, g in d.groupby("域"):
            v = g["net20"]
            out.append(f"| {NAME(u_)} | {len(v)} | {v.max()*100:+.2f}% | {v.median()*100:+.2f}% | "
                       f"{v.quantile(.25)*100:+.2f}% | {v.min()*100:+.2f}% | "
                       f"**{(v>0).mean()*100:.0f}%** |")
    out += ["", "**最优值不可作为预期**:它是在验证段上从数十个配置里搜出来的,含多重检验膨胀。",
            "中位数更接近「随便挑一个合理配置」的预期,为正占比说明结论对配置选择的敏感性。", ""]

    out += ["## ④ 多种子分布", "", "见 `多种子分布.md`(需重训 20 个单种子模型,单独产出)。", ""]

    with open(os.path.join(RESULTS, "稳健性检验.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    bt.to_csv(os.path.join(RESULTS, "稳健性_自助.csv"), index=False)
    print("\n".join(out))


if __name__ == "__main__":
    main()
