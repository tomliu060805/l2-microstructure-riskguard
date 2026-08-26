# -*- coding: utf-8 -*-
"""A1 风格中性化检验 — 超额是真选股还是风格暴露?

## 为什么这是最高优先级
诊断显示预测最差组的日内波动 317bp vs 最好组 181bp —— 最差组确实更高波动。
若超额主要来自"躲开高波动小盘股"这个**风格暴露**, 那它不是选股 alpha,
而是可以用更便宜的方式(直接做风格因子)获得的东西。这是唯一一个
**可能推翻整个项目结论**的检验。

## 风格因子(全部只用决策时点已知的信息)
  size    log(流通市值), t−1 日口径
  vol20   过去20日已实现波动
  beta20  过去20日对域内等权收益的beta
  mom20   过去20日累计收益(中期动量)
  rev5    过去5日累计收益(**短期反转** — 微观结构因子最可能的混淆项)

## 方法
逐日横截面 OLS: score ~ 1 + styles, 取**残差**作为中性化后分数, 重跑同一回测。
另报 ①回归R²(分数有多少能被风格解释) ②被剔除组合的风格暴露(在赌什么)。

用法: /usr/bin/python3 style_neutralize.py [--index hs300 csi500 csi1000 gz2000]
"""
import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

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
from short_side_exclude import build, simulate, ANN
from dsutil import ds_dir, pred_dir

warnings.filterwarnings("ignore")

STYLES = ["size", "vol20", "beta20", "mom20", "rev5"]
PRED_DEFAULT = "lgbm_y5d_rank"      # 平台四域主模型均为对称回归


def NAME(idx):
    return C.INDEXES[idx]["name"]


def load_mcap(date):
    fp = os.path.join("${IDXML_MCAP_ROOT}", f"{date}.parquet")
    if not os.path.exists(fp):
        return None
    d = pd.read_parquet(fp, columns=["code", "circulating_market_cap"])
    d = d.rename(columns={"code": "symbol", "circulating_market_cap": "mcap"})
    d["date"] = date
    return d


def build_styles(ds_dir, dates_needed):
    """从收益历史构建风格因子. 全部用 t−1 及更早的信息 → 无前视."""
    ds = pd.read_parquet(os.path.join(ds_dir, "ds.parquet"),
                         columns=["date", "symbol", "ret_fwd_1d"])
    ds["date"] = ds["date"].astype(str)
    ds = ds.sort_values(["symbol", "date"])
    g = ds.groupby("symbol")["ret_fwd_1d"]
    # ret_fwd_1d(t) 是 t→t+1 收益; 决策点 t 已知的最近一个完整日收益是 ret_fwd_1d(t−1)
    ds["r_lag"] = g.shift(1)
    gl = ds.groupby("symbol")["r_lag"]
    ds["vol20"] = gl.transform(lambda s: s.rolling(20, min_periods=10).std())
    ds["mom20"] = gl.transform(lambda s: s.rolling(20, min_periods=10).sum())
    ds["rev5"] = gl.transform(lambda s: s.rolling(5, min_periods=3).sum())
    # beta: 对域内等权收益
    mkt = ds.groupby("date")["r_lag"].transform("mean")
    ds["_x"] = mkt
    ds["_xy"] = ds["r_lag"] * mkt
    ds["_xx"] = mkt * mkt
    gg = ds.groupby("symbol")
    cov = gg["_xy"].transform(lambda s: s.rolling(20, min_periods=10).mean()) - \
        gg["r_lag"].transform(lambda s: s.rolling(20, min_periods=10).mean()) * \
        gg["_x"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    var = gg["_xx"].transform(lambda s: s.rolling(20, min_periods=10).mean()) - \
        gg["_x"].transform(lambda s: s.rolling(20, min_periods=10).mean()) ** 2
    ds["beta20"] = cov / var.replace(0, np.nan)

    # 市值: 用 t−1 日口径(t日收盘市值含当日收盘价, 决策点14:55严格说未知)
    alldates = sorted(set(ds["date"]) & set(dates_needed) | set(dates_needed))
    with ProcessPoolExecutor(30) as ex:
        mc = [r for r in ex.map(load_mcap, sorted(set(dates_needed))) if r is not None]
    mc = pd.concat(mc, ignore_index=True)
    mc = mc.sort_values(["symbol", "date"])
    mc["mcap_lag"] = mc.groupby("symbol")["mcap"].shift(1)
    mc["size"] = np.log(mc["mcap_lag"].clip(lower=1e-6))
    out = ds[["date", "symbol", "vol20", "beta20", "mom20", "rev5"]].merge(
        mc[["date", "symbol", "size"]], on=["date", "symbol"], how="left")
    return out


def neutralize(df, score_col="score"):
    """逐日横截面 OLS 取残差; 返回 (中性化分数, 平均R²)."""
    resid, r2s = [], []
    for d, g in df.groupby("date", sort=True):
        y = g[score_col].values.astype(float)
        X = g[STYLES].values.astype(float)
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        r = pd.Series(np.nan, index=g.index)
        if ok.sum() > len(STYLES) + 30:
            Xs = X[ok]
            # 风格先做横截面标准化, 避免量纲主导
            Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12)
            A = np.column_stack([np.ones(ok.sum()), Xs])
            yy = y[ok] - y[ok].mean()
            coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
            fit = A @ coef
            r.iloc[np.where(ok)[0]] = yy - fit
            ss = float(((yy - fit) ** 2).sum()), float((yy ** 2).sum())
            r2s.append(1 - ss[0] / (ss[1] + 1e-12))
        resid.append(r)
    return pd.concat(resid).sort_index(), float(np.mean(r2s)) if r2s else np.nan


def exposure(df, frac=0.30, score_col="score"):
    """被剔除组合的风格暴露(标准化后, 剔除组均值 − 全域均值)."""
    rows = []
    for d, g in df.groupby("date", sort=True):
        g = g.dropna(subset=[score_col])
        if len(g) < 50:
            continue
        cut = g[score_col].quantile(frac)
        drop = g[score_col] <= cut
        z = (g[STYLES] - g[STYLES].mean()) / (g[STYLES].std() + 1e-12)
        rows.append(z[drop.values].mean())
    return pd.DataFrame(rows).mean()


def run(index, pred=PRED_DEFAULT):
    df, dates = build(index, pred_name=pred)
    st = build_styles(ds_dir(index), dates)
    df = df.merge(st, on=["date", "symbol"], how="left")
    cov = df[STYLES].notna().all(1).mean()
    print(f"  风格覆盖率 {cov:.3f}", flush=True)

    df["score_neu"], r2 = neutralize(df)
    expo = exposure(df, score_col="score")

    out = {}
    for tag, col in [("原始", "score"), ("中性化", "score_neu")]:
        d2 = df.sort_values(["symbol", "date"]).copy()
        d2["score_sm"] = d2.groupby("symbol")[col].transform(
            lambda s: s.rolling(5, min_periods=1).mean())
        d2 = d2.sort_values(["date", "symbol"]).reset_index(drop=True)
        res, te, tw, nd = simulate(d2, 0.30, "smooth")
        out[tag] = dict(超额毛=res[0]["ann"], 净20bp=res[20]["ann"], IR=res[20]["ir"],
                        TE=te, 换手=tw, 胜率=res[20]["wr"], 盈利因子=res[20]["pf"])
        print(f"  {tag}: 净20bp {res[20]['ann']*100:+.2f}% IR {res[20]['ir']:.2f}", flush=True)
    return out, r2, expo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", nargs="*", default=C.ALL_INDEXES)
    ap.add_argument("--pred", default=PRED_DEFAULT)
    args = ap.parse_args()

    allr = {}
    for u in args.index:
        if not os.path.exists(os.path.join(ds_dir(u), "ds.parquet")):
            print(f"跳过 {u}(无数据集)", flush=True)
            continue
        if not os.path.exists(os.path.join(pred_dir(u), f"{args.pred}.parquet")):
            print(f"跳过 {u}(无预测 {args.pred})", flush=True)
            continue
        print(f"=== {u}", flush=True)
        allr[u] = run(u, args.pred)

    out = ["# A1 风格中性化检验 — 超额是真选股还是风格暴露?", "",
           f"> 验证段 2022-07-01~2024-06-28 | 信号={args.pred} | "
           "落地形态=5日平滑+剔除底部30% | **测试段未触碰**", "",
           "**风格因子**(全部只用决策时点已知信息): size=log流通市值(t−1口径)、"
           "vol20=过去20日已实现波动、beta20=过去20日对域内等权beta、"
           "mom20=过去20日累计收益、**rev5=过去5日累计收益(短期反转,"
           "微观结构因子最可能的混淆项)**。", "",
           "**方法**: 逐日横截面 OLS `score ~ 1 + 标准化风格`, 取**残差**重跑同一回测。", "",
           "## 中性化前后对比", "",
           "| 域 | 口径 | 超额毛 | 超额净20bp | IR | TE | 换手 | 盈利因子 |",
           "|---|---|---|---|---|---|---|---|"]
    for u, (r, r2, expo) in allr.items():
        for tag in ["原始", "中性化"]:
            v = r[tag]
            out.append(f"| {NAME(u)} | {tag} | {v['超额毛']*100:+.2f}% | "
                       f"**{v['净20bp']*100:+.2f}%** | **{v['IR']:.2f}** | {v['TE']*100:.2f}% | "
                       f"{v['换手']:.3f} | {v['盈利因子']:.2f} |")
    out += ["", "## 分数被风格解释的比例(逐日横截面 R²)", "",
            "| 域 | 平均 R² | 解读 |", "|---|---|---|"]
    for u, (r, r2, expo) in allr.items():
        note = "分数基本独立于风格" if r2 < 0.15 else ("有一定风格成分" if r2 < 0.35 else "风格成分很重")
        out.append(f"| {NAME(u)} | {r2:.3f} | {note} |")

    out += ["", "## 被剔除组合的风格暴露(标准化后, 剔除组均值 − 全域均值)", "",
            "| 域 | " + " | ".join(STYLES) + " |", "|---|" + "---|" * len(STYLES)]
    for u, (r, r2, expo) in allr.items():
        out.append(f"| {NAME(u)} | " + " | ".join(f"{expo[s]:+.3f}" for s in STYLES) + " |")
    out += ["", "负值 = 剔除组在该风格上低于全域均值。例如 size 为负 = 倾向剔除小市值股;",
            "rev5 为负 = 倾向剔除近期已下跌的股票(动量延续而非反转)。", ""]

    # 留存率
    out += ["## 判定: 超额留存率", "", "| 域 | 原始净20bp | 中性化后 | **留存率** |", "|---|---|---|---|"]
    for u, (r, r2, expo) in allr.items():
        a, b = r["原始"]["净20bp"], r["中性化"]["净20bp"]
        out.append(f"| {NAME(u)} | {a*100:+.2f}% | {b*100:+.2f}% | **{b/a*100:.0f}%** |")
    out.append("")

    txt = "\n".join(out) + "\n"
    for u in allr:
        with open(os.path.join(results_dir(u), "A1_风格中性化检验.md"), "w") as f:
            f.write(txt)
    with open(os.path.join(RESULTS, "A1_风格中性化检验_四域.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
