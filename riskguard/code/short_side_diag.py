# -*- coding: utf-8 -*-
"""空头端规避可行性诊断(验证段, 不训练新模型).

回答三个能杀死这个方向的问题:
  A. 权重稀释 — 预测最差的票在**指数市值加权**口径下占多少权重?
  B. 跌停卖不掉 — 最该躲的票有多少在决策时点已封跌停?可规避的亏损有多少?
  C. 风格伪装 — D1是不是只是"小盘+高波动"?
另出 D. 上下尾信号强度分解, E. 剔除底部K%的指增模拟(等权与指数权重两口径)。

用法: IDXML_DS_ROOT=../../data /usr/bin/python3 short_side_diag.py
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
RESULTS = os.path.join(HERE, "..", "results")
import common as C
from dsutil import ds_dir, pred_dir


def results_dir(index):
    d = os.path.join(RESULTS, index)
    os.makedirs(d, exist_ok=True)
    return d
from btutil import tie_broken_decile, NQ

PRED = "lgbm_y5d_rank"
T1455 = pd.Timestamp("2000-01-01 14:55:00").time()


def load_limit(date):
    fp = os.path.join(C.LIMIT_ROOT, f"{date}.parquet")
    if not os.path.exists(fp):
        return None
    d = pd.read_parquet(fp)
    d = d[d["datetime"].dt.time == T1455]
    d = d[["symbol", "limit_status"]].copy()
    d["date"] = date
    return d


def load_mcap(date):
    fp = os.path.join("${IDXML_MCAP_ROOT}", f"{date}.parquet")
    if not os.path.exists(fp):
        return None
    d = pd.read_parquet(fp, columns=["code", "circulating_market_cap"])
    d = d.rename(columns={"code": "symbol"})
    d["date"] = date
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True,
                    help="hs300/csi500/csi1000/gz2000")
    args = ap.parse_args()
    globals()["args"] = args    # 模块级函数(load_limit等)也要用

    pred = pd.read_parquet(os.path.join(pred_dir(args.index), f"{PRED}.parquet"))
    pred["date"] = pred["date"].astype(str)
    ds = pd.read_parquet(os.path.join(ds_dir(args.index), "ds.parquet"),
                         columns=["date", "symbol", "ret_fwd_1d", "tradable"])
    ds["date"] = ds["date"].astype(str)
    df = pred.merge(ds, on=["date", "symbol"], how="inner")
    df = df[(df["date"] >= C.VAL_START) & (df["date"] < C.TEST_START)]
    dates = sorted(df["date"].unique())
    print(f"验证段 {len(dates)} 天, {len(df)} 行", flush=True)

    # 指数权重
    w = C.load_weights([args.index]).rename(columns={"symbol": "stock_code"})
    w["date"] = w["date"].astype(str)
    w = w[w["date"].isin(dates)].copy()
    w["symbol"] = w["stock_code"].str[:6]
    w = w[["date", "symbol", "weight"]]
    df = df.merge(w, on=["date", "symbol"], how="left")
    print(f"权重匹配率 {df['weight'].notna().mean():.3f}", flush=True)

    # 涨跌停 + 市值
    with ProcessPoolExecutor(40) as ex:
        lim = [r for r in ex.map(load_limit, dates) if r is not None]
        mc = [r for r in ex.map(load_mcap, dates) if r is not None]
    lim = pd.concat(lim, ignore_index=True)
    df = df.merge(lim, on=["date", "symbol"], how="left")
    if mc:
        mc = pd.concat(mc, ignore_index=True)
        df = df.merge(mc, on=["date", "symbol"], how="left")
        print(f"市值覆盖 {df['circulating_market_cap'].notna().mean():.3f}", flush=True)
    else:
        df["circulating_market_cap"] = np.nan

    # 分组
    parts = []
    for d, g in df.groupby("date", sort=True):
        g = g.dropna(subset=["score"]).copy()
        g["q"] = tie_broken_decile(g["score"].values, d)
        g["w_norm"] = g["weight"] / g["weight"].sum()
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)

    out = ["# 空头端规避 — 可行性诊断", "",
           f"> 验证段 {dates[0]}~{dates[-1]}, {len(dates)}天 | 信号={PRED} | 测试段未触碰", ""]

    # ---------- A. 权重稀释 ----------
    out += ["## A. 权重稀释: 预测最差的票占多少指数权重?", "",
            "| 十分组 | 平均指数权重合计 | 等权口径次日收益(年化) | 权重口径次日收益(年化) |",
            "|---|---|---|---|"]
    ann = 242
    for k in range(NQ):
        sub = df[df["q"] == k]
        wsum = sub.groupby("date")["w_norm"].sum().mean()
        eq = sub.groupby("date")["ret_fwd_1d"].mean().mean() * ann
        wt = sub.groupby("date").apply(
            lambda g: np.nansum(g["w_norm"] * g["ret_fwd_1d"]) / max(g["w_norm"].sum(), 1e-12)).mean() * ann
        out.append(f"| D{k+1} | {wsum*100:.1f}% | {eq*100:+.1f}% | {wt*100:+.1f}% |")
    d1w = df[df["q"] == 0].groupby("date")["w_norm"].sum().mean()
    out += ["", f"**D1(预测最差10%)平均只占指数权重 {d1w*100:.1f}%** "
            f"(等权下是10%) → 完全剔除D1最多改变组合 {d1w*100:.1f}% 的仓位。", ""]

    # ---------- B. 跌停卖不掉 ----------
    out += ["## B. 跌停卖不掉: 最该躲的票躲得掉吗?", ""]
    d1 = df[df["q"] == 0]
    dn = (d1["limit_status"] == -1).mean()
    up = (d1["limit_status"] == 1).mean()
    untr = (d1["tradable"] == 0).mean()
    r_all = d1.groupby("date")["ret_fwd_1d"].mean().mean() * ann
    r_tr = d1[d1["tradable"] == 1].groupby("date")["ret_fwd_1d"].mean().mean() * ann
    r_un = d1[d1["tradable"] == 0].groupby("date")["ret_fwd_1d"].mean().mean() * ann
    out += [f"- D1中决策时点(14:55)**封跌停**的比例: **{dn*100:.2f}%**; 封涨停 {up*100:.2f}%; "
            f"不可交易(涨跌停/停牌)合计 {untr*100:.2f}%",
            f"- D1整体次日年化 {r_all*100:+.1f}%; 其中**可交易部分 {r_tr*100:+.1f}%**, "
            f"不可交易部分 {r_un*100:+.1f}%",
            "", f"→ **可规避的那部分(可交易) {r_tr*100:+.1f}%** 才是能真正抓到的; "
            f"若可交易部分收益明显弱于整体, 说明信号的强度有相当比例锁在躲不掉的票里。", ""]

    # ---------- C. 风格伪装 ----------
    out += ["## C. 风格伪装: D1是不是只是小盘+高波?", "",
            "| 十分组 | 流通市值中位数(亿) | 次日收益标准差(bp) |", "|---|---|---|"]
    for k in [0, 1, 4, 8, 9]:
        sub = df[df["q"] == k]
        mcm = sub["circulating_market_cap"].median()
        vol = sub["ret_fwd_1d"].std() * 1e4
        mcm_s = f"{mcm:.0f}" if pd.notna(mcm) else "—"
        out.append(f"| D{k+1} | {mcm_s} | {vol:.0f} |")
    out.append("")

    # ---------- D. 上下尾强度 ----------
    uni = df.groupby("date")["ret_fwd_1d"].mean()
    d1_ex = (df[df["q"] == 0].groupby("date")["ret_fwd_1d"].mean() - uni).mean() * ann
    d10_ex = (df[df["q"] == 9].groupby("date")["ret_fwd_1d"].mean() - uni).mean() * ann
    out += ["## D. 上下尾信号强度(相对全域等权基准的超额)", "",
            f"- **下尾 D1: {d1_ex*100:+.1f}%/年** (规避它可获得的等权超额)",
            f"- 上尾 D10: {d10_ex*100:+.1f}%/年",
            f"- 下尾/上尾强度比 = **{abs(d1_ex/d10_ex):.2f}x**", ""]

    # ---------- E. 剔除底部K% 的指增模拟 ----------
    out += ["## E. 剔除底部K%: 指增模拟(每日调仓, 只在可交易时剔除)", "",
            "### E1. 指数权重口径(真实指增)", "",
            "| 剔除比例 | 超额毛 | 超额净20bp | IR(净) | TE | 单边日换手 | 胜率 |",
            "|---|---|---|---|---|---|---|"]

    def simulate(frac, weight_col):
        """按 weight_col 加权; 剔除底部frac(仅当可交易), 权重按比例再分配."""
        prev = {}
        ex_r, tw_l = [], []
        for d, g in df.groupby("date", sort=True):
            g = g.dropna(subset=[weight_col, "ret_fwd_1d"])
            if len(g) < 50:
                continue
            wb = g[weight_col].values / g[weight_col].sum()
            n = len(g)
            kcut = int(np.ceil(n * frac))
            order = np.argsort(g["score"].values)
            drop_idx = order[:kcut]
            can = g["tradable"].values == 1
            mask = np.ones(n, bool)
            mask[drop_idx] = False
            mask |= ~can          # 不可交易 → 躲不掉, 保持基准权重
            wp = wb * mask
            wp = wp / wp.sum()
            r = g["ret_fwd_1d"].values
            ex_r.append(np.nansum(wp * r) - np.nansum(wb * r))
            syms = g["symbol"].values
            cur = dict(zip(syms, wp))
            tw_l.append(sum(abs(cur.get(s, 0) - prev.get(s, 0)) for s in set(cur) | set(prev)))
            prev = cur
        ex_r = np.array(ex_r)
        tw = np.array(tw_l)
        te = ex_r.std() * np.sqrt(ann)
        res = {}
        for bp in (0, 20):
            v = ex_r - tw * bp / 1e4
            res[bp] = dict(ann=v.mean() * ann, ir=v.mean() / (v.std() + 1e-12) * np.sqrt(ann),
                           wr=(v > 0).mean())
        return res, te, tw.mean()

    for frac in (0.05, 0.10, 0.20, 0.30):
        res, te, tw = simulate(frac, "w_norm")
        out.append(f"| 底部{frac*100:.0f}% | {res[0]['ann']*100:+.2f}% | "
                   f"**{res[20]['ann']*100:+.2f}%** | **{res[20]['ir']:.2f}** | "
                   f"{te*100:.2f}% | {tw:.3f} | {res[20]['wr']*100:.0f}% |")

    out += ["", "### E2. 等权口径(对照: 若产品做等权增强)", "",
            "| 剔除比例 | 超额毛 | 超额净20bp | IR(净) | TE | 单边日换手 | 胜率 |",
            "|---|---|---|---|---|---|---|"]
    df["w_eq"] = 1.0
    for frac in (0.05, 0.10, 0.20, 0.30):
        res, te, tw = simulate(frac, "w_eq")
        out.append(f"| 底部{frac*100:.0f}% | {res[0]['ann']*100:+.2f}% | "
                   f"**{res[20]['ann']*100:+.2f}%** | **{res[20]['ir']:.2f}** | "
                   f"{te*100:.2f}% | {tw:.3f} | {res[20]['wr']*100:.0f}% |")

    txt = "\n".join(out) + "\n"
    fp = os.path.join(results_dir(args.index), "空头端规避_诊断.md")
    with open(fp, "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
