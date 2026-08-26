# -*- coding: utf-8 -*-
"""成本进目标函数: 用无交易带(no-trade band)替代事后平滑 + 固定剔除比例.

## 动机
现有最优配置有两个"拍脑袋"参数: 剔除比例K%(=30%) 与平滑窗口(=5日)。
两者都是为压换手而事后加的补丁, 面试时会被质疑"是不是为了让成本后好看而调出来的"。
本脚本把成本直接写进目标函数, 让**剔除多少、何时换**由经济性内生决定。

## 形式化
每日对每只股票选持有比例 x_i ∈ [0,1](1=按基准权重持有, 0=完全剔除),
组合权重 w_i ∝ wb_i · x_i。目标(忽略再归一化的二阶项):

    max_x  Σ wb_i·x_i·α_i  −  λ·Σ wb_i·|x_i − x_prev,i|

α_i = 预期未来5日超额收益(收益单位), λ = 单边成本率。
两项对 i 可分离, 且目标关于 x_i 线性 → 最优解在顶点, 逐股比较三个候选值可得**闭式解**:

    α_i < −λ  → 剔除(x=0)      α_i > +λ  → 恢复持有(x=1)      否则保持不动

即一条 **[−λ, +λ] 的无交易带**。带宽由成本直接决定, 不是调出来的。
成本越高带越宽 → 换手自动下降; 剔除只数随信号强弱内生浮动。

## α 的标定(关键, 且必须无前视)
模型分数是秩/概率, 不是收益单位, 必须标定成"预期5日超额收益"。
做法: 在每个 refit 边界重新标定, 只用 **该边界之前(再减embargo)** 的**样本外**预测,
按分数百分位分20档算已实现5日超额收益均值, 线性插值。
→ 标定期与被评估期严格不重叠, 与模型refit同频。

用法: IDXML_DS_ROOT=../../data /usr/bin/python3 cost_aware_exclude.py
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

PRED = "tail_clf_w_y5d"     # 前身项目最优模型(深跌加权二分类)
INDEX = os.environ.get("IDXML_INDEX", "csi1000")   # build() 等模块级函数用
N_BINS = 20
RECALIB_EVERY = 63          # 与模型 refit 同频
EMBARGO = C.EMBARGO_DAYS


def add_fwd5(df):
    r5 = pd.read_parquet(os.path.join(ds_dir(INDEX), "ds.parquet"),
                         columns=["date", "symbol", "ret_fwd_5d"])
    r5["date"] = r5["date"].astype(str)
    return df.merge(r5, on=["date", "symbol"], how="left")


def calibrate(hist):
    """hist: 含 score_pct 与 ex5(已实现5日超额收益) 的历史样本外数据.
    返回 (bin_edges, bin_mean) 供插值; 无数据时返回 None."""
    h = hist.dropna(subset=["score_pct", "ex5"])
    if len(h) < 20000:
        return None
    edges = np.linspace(0, 1, N_BINS + 1)
    idx = np.clip(np.digitize(h["score_pct"].values, edges[1:-1]), 0, N_BINS - 1)
    means = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        m = idx == b
        if m.sum() > 200:
            means[b] = h["ex5"].values[m].mean()
        # 单调化(等渗近似): 分数越低预期超额越低
    ok = ~np.isnan(means)
    if ok.sum() < 5:
        return None
    means = np.interp(np.arange(N_BINS), np.arange(N_BINS)[ok], means[ok])
    means = np.maximum.accumulate(means)      # 强制单调不减
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, means


def apply_calib(calib, pct):
    centers, means = calib
    return np.interp(pct, centers, means)


def simulate_band(df, lam_bps, calibs, center="portfolio", max_excl=None,
                  cost_bps=(0, 10, 20, 30)):
    """无交易带.

    center='portfolio'(正确): 剔除i时其权重再分配给现有持仓, 故机会成本是
        组合平均α, 判据 α_i < ᾱ_held − λ / α_i > ᾱ_held + λ。
    center='zero'(朴素, 已证伪): 以0为中心 → 因标定曲线正侧扁平(多数股α≈+10bp),
        剔除易、恢复难, 形成**棘轮**: 剔除比例484天内从40%单调爬到70%, TE失控。
    max_excl: 剔除比例上限(指增授权约束), 超出时只保留α最低的那批。
    """
    lam = lam_bps / 1e4
    prev_w, held_state = {}, {}
    ex_r, tw_l, n_excl = [], [], []
    for d, g in df.groupby("date", sort=True):
        g = g.dropna(subset=["weight", "ret_fwd_1d", "score_pct"])
        if len(g) < 50:
            continue
        calib = calibs.get(d)
        if calib is None:
            continue
        syms = g["symbol"].values
        wb = g["weight"].values / g["weight"].sum()
        alpha = apply_calib(calib, g["score_pct"].values)
        can = g["tradable"].values == 1
        held = np.array([held_state.get(s, 1) for s in syms], np.int8)

        if center == "portfolio":
            hw = wb * held
            c0 = float((hw * alpha).sum() / hw.sum()) if hw.sum() > 0 else 0.0
        else:
            c0 = 0.0
        new = held.copy()
        new[(held == 1) & (alpha < c0 - lam)] = 0    # 相对组合的机会成本盖过交易成本
        new[(held == 0) & (alpha > c0 + lam)] = 1
        new[~can] = held[~can]                        # 涨跌停/停牌 → 动不了
        if max_excl is not None:
            # 授权上限用**增量式**维护: 尽量保留已有排除集, 只把超出部分中
            # α最高(最没必要排除)的那几只恢复回来。整批重挑会破坏迟滞→换手爆炸。
            cap_n = int(len(new) * max_excl)
            excl_idx = np.where(new == 0)[0]
            if len(excl_idx) > cap_n:
                order = excl_idx[np.argsort(-alpha[excl_idx])]
                need = len(excl_idx) - cap_n
                for i in order:
                    if need <= 0:
                        break
                    if can[i]:
                        new[i] = 1
                        need -= 1
        held_state.update(dict(zip(syms, new)))

        wp = wb * new
        if wp.sum() <= 0:
            continue
        wp = wp / wp.sum()
        n_excl.append(float((new == 0).mean()))

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
    res = {}
    for bp in cost_bps:
        v = ex_r - tw * bp / 1e4
        win, loss = v[v > 0], v[v < 0]
        res[bp] = dict(ann=v.mean() * ANN, ir=v.mean() / (v.std() + 1e-12) * np.sqrt(ANN),
                       wr=float((v > 0).mean()),
                       pf=float(win.sum() / abs(loss.sum())) if len(loss) else np.nan)
    return res, te, tw.mean(), float(np.mean(n_excl)), len(ex_r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True,
                    help="hs300/csi500/csi1000/gz2000")
    args = ap.parse_args()
    globals()["args"] = args    # build() 等模块级函数也要用
    SSE.args = args             # 被导入模块 short_side_exclude 也要用

    base, dates = build()
    p = pd.read_parquet(os.path.join(pred_dir(args.index), f"{PRED}.parquet"))
    p["date"] = p["date"].astype(str)

    # 全量(含val之前)样本外预测, 供标定使用
    full = p.merge(base[["date", "symbol"]].drop_duplicates(), on=["date", "symbol"], how="outer")
    df = base.drop(columns=["score", "score_sm"]).merge(p, on=["date", "symbol"], how="inner")
    df = add_fwd5(df)
    df["score_pct"] = df.groupby("date")["score"].rank(pct=True)

    # 标定用的历史池: 全部样本外预测(2018-01起), 含 val 之前的 dev 段
    hist_all = pd.read_parquet(os.path.join(ds_dir(args.index), "ds.parquet"),
                               columns=["date", "symbol", "ret_fwd_5d"])
    hist_all["date"] = hist_all["date"].astype(str)
    hist = p.merge(hist_all, on=["date", "symbol"], how="inner")
    hist["score_pct"] = hist.groupby("date")["score"].rank(pct=True)
    hist["ex5"] = hist["ret_fwd_5d"] - hist.groupby("date")["ret_fwd_5d"].transform("mean")
    hist_dates = np.array(sorted(hist["date"].unique()))

    # 每 RECALIB_EVERY 天重标定一次, 只用边界前(减embargo)的历史
    val_dates = sorted(df["date"].unique())
    calibs, cur, last_anchor = {}, None, None
    for i, d in enumerate(val_dates):
        if i % RECALIB_EVERY == 0:
            cut_i = np.searchsorted(hist_dates, d) - EMBARGO
            if cut_i > 0:
                cur = calibrate(hist[hist["date"] < hist_dates[cut_i]])
                last_anchor = d
        calibs[d] = cur
    n_ok = sum(v is not None for v in calibs.values())
    print(f"标定完成: {n_ok}/{len(val_dates)} 天有可用标定曲线", flush=True)
    if cur is not None:
        centers, means = cur
        print("最后一条标定曲线(分数百分位→预期5日超额, bp):", flush=True)
        print("  " + "  ".join(f"{c:.2f}:{m*1e4:+.0f}" for c, m in zip(centers[::4], means[::4])), flush=True)

    out = ["# 成本进目标函数 — 无交易带 vs 事后平滑", "",
           f"> 验证段 {val_dates[0]}~{val_dates[-1]}, {len(val_dates)}天 | 信号={PRED}"
           f" | 基准={C.INDEXES[args.index]['name']}指数权重 | **测试段未触碰**", "",
           "**做法**: 把成本写进目标函数 `max Σwb·x·α − λ·Σwb·|x−x_prev|`, 逐股可分离,",
           "闭式解是一条 **[−λ,+λ] 无交易带**: α<−λ 剔除、α>+λ 恢复、其间不动。",
           "带宽由成本 λ 直接决定, **剔除比例内生**(不再是拍脑袋的K%), 也不需要事后平滑。",
           "α 由分数百分位标定成预期5日超额收益, 每63天用**边界前的样本外数据**重标定(无前视)。", "",
           "## A. 无交易带(λ 即假设的单边成本)", "",
           "| 带宽λ | 平均剔除比例 | 超额毛 | 净10bp | 净20bp | 净30bp | IR(净20) | TE | 换手 | 胜率 | 盈利因子 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lam in (5, 10, 20, 30, 50, 80):
        res, te, tw, nex, nd = simulate_band(df, lam, calibs, center="portfolio")
        r20 = res[20]
        out.append(f"| {lam}bp | {nex*100:.1f}% | {res[0]['ann']*100:+.2f}% | "
                   f"{res[10]['ann']*100:+.2f}% | **{res[20]['ann']*100:+.2f}%** | "
                   f"{res[30]['ann']*100:+.2f}% | **{r20['ir']:.2f}** | {te*100:.2f}% | "
                   f"{tw:.3f} | {r20['wr']*100:.0f}% | {r20['pf']:.2f} |")

    out += ["", "## A2. 加授权上限(剔除≤30%, 模拟指增的偏离约束)", "",
            "| 带宽λ | 平均剔除比例 | 超额毛 | 净20bp | IR | TE | 换手 |",
            "|---|---|---|---|---|---|---|"]
    for lam in (5, 10, 20, 30):
        res, te, tw, nex, nd = simulate_band(df, lam, calibs, center="portfolio", max_excl=0.30)
        out.append(f"| {lam}bp | {nex*100:.1f}% | {res[0]['ann']*100:+.2f}% | "
                   f"**{res[20]['ann']*100:+.2f}%** | **{res[20]['ir']:.2f}** | "
                   f"{te*100:.2f}% | {tw:.3f} |")

    out += ["", "## A3. 朴素版(band以0为中心) — **已证伪, 留作反例**", "",
            "剔除比例484天内从40%单调爬到70%(棘轮), TE失控到7.7%, 已不是规避而是集中选股。", "",
            "| 带宽λ | 平均剔除比例 | 超额毛 | 净20bp | IR | TE | 换手 |",
            "|---|---|---|---|---|---|---|"]
    for lam in (10, 20):
        res, te, tw, nex, nd = simulate_band(df, lam, calibs, center="zero")
        out.append(f"| {lam}bp | {nex*100:.1f}% | {res[0]['ann']*100:+.2f}% | "
                   f"{res[20]['ann']*100:+.2f}% | {res[20]['ir']:.2f} | "
                   f"{te*100:.2f}% | {tw:.3f} |")

    out += ["", "## B. 对照: 事后平滑 + 固定剔除比例(现有最优配置)", "",
            "| 配置 | 剔除比例 | 超额毛 | 净20bp | IR | TE | 换手 |", "|---|---|---|---|---|---|---|"]
    df2 = df.sort_values(["symbol", "date"]).copy()
    df2["score_sm"] = df2.groupby("symbol")["score"].transform(
        lambda s: s.rolling(5, min_periods=1).mean())
    df2 = df2.sort_values(["date", "symbol"]).reset_index(drop=True)
    for frac in (0.10, 0.30):
        res, te, tw, nd = simulate(df2, frac, "smooth")
        out.append(f"| 5日平滑+剔除{frac*100:.0f}% | {frac*100:.0f}%(固定) | "
                   f"{res[0]['ann']*100:+.2f}% | **{res[20]['ann']*100:+.2f}%** | "
                   f"**{res[20]['ir']:.2f}** | {te*100:.2f}% | {tw:.3f} |")
    out.append("")

    txt = "\n".join(out) + "\n"
    with open(os.path.join(results_dir(args.index), "成本进目标函数_无交易带.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
