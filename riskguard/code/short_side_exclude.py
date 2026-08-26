# -*- coding: utf-8 -*-
"""空头端规避 v2: 换手控制下的"剔除底部"指增回测.

对 v1 诊断的两处修正 + 三种换手控制:
  修正1 换手口径 — 基准权重每日随股价漂移, 持仓自动跟随不需交易;
        换手须用 |w_t − w_{t−1}漂移后| 而非 |w_t − w_{t−1}|(否则系统性高估成本);
  修正2 基准自身再平衡(成分调整/权重更新)不计入主动换手;
  控制A plain   — 每日按当日分数剔除底部K%
  控制B buffer  — 进入排除名单需底部K%, 退出需回到底部K*B%以上(迟滞, 防边界抖动)
  控制C smooth  — 用过去N日分数均值决定(信号平滑, 与5日标签期限匹配)

用法: IDXML_DS_ROOT=../../data /usr/bin/python3 short_side_exclude.py
"""
import argparse
import os
import sys
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
from dsutil import ds_dir, pred_dir

PRED = "lgbm_y5d_rank"
ANN = 242
T1455 = pd.Timestamp("2000-01-01 14:55:00").time()


def load_limit(date):
    fp = os.path.join(C.LIMIT_ROOT, f"{date}.parquet")
    if not os.path.exists(fp):
        return None
    d = pd.read_parquet(fp)
    d = d[d["datetime"].dt.time == T1455][["symbol", "limit_status"]].copy()
    d["date"] = date
    return d


def build(index=None, pred_name=None, window="val"):
    """index/pred_name 可显式传入(供 A1/A2/稳健性等分析脚本复用);
    不传时回落到命令行 args(保持原有调用方式不变)。"""
    index = index or args.index
    pred_name = pred_name or PRED
    pred = pd.read_parquet(os.path.join(pred_dir(index), f"{pred_name}.parquet"))
    pred["date"] = pred["date"].astype(str)
    ds = pd.read_parquet(os.path.join(ds_dir(index), "ds.parquet"),
                         columns=["date", "symbol", "ret_fwd_1d", "tradable"])
    ds["date"] = ds["date"].astype(str)
    df = pred.merge(ds, on=["date", "symbol"], how="inner")
    if window == "test":
        df = df[df["date"] >= C.TEST_START]          # 测试段一次性评估
    else:
        df = df[(df["date"] >= C.VAL_START) & (df["date"] < C.TEST_START)]
    dates = sorted(df["date"].unique())

    w = C.load_weights([index]).rename(columns={"symbol": "stock_code"})
    w["date"] = w["date"].astype(str)
    w = w[w["date"].isin(dates)].copy()
    w["symbol"] = w["stock_code"].str[:6]
    df = df.merge(w[["date", "symbol", "weight"]], on=["date", "symbol"], how="left")

    with ProcessPoolExecutor(40) as ex:
        lim = [r for r in ex.map(load_limit, dates) if r is not None]
    df = df.merge(pd.concat(lim, ignore_index=True), on=["date", "symbol"], how="left")

    # 平滑分数(过去5日均值, 仅用历史 → 无前视)
    df = df.sort_values(["symbol", "date"])
    df["score_sm"] = (df.groupby("symbol")["score"]
                      .transform(lambda s: s.rolling(5, min_periods=1).mean()))
    return df.sort_values(["date", "symbol"]).reset_index(drop=True), dates


def simulate(df, frac, mode, weight_col="weight", buf_mult=2.0, cost_bps=(0, 10, 20, 30)):
    """剔除底部frac; 换手按漂移调整后的主动交易量计。"""
    prev_w = {}          # 上一日收盘后的实际持仓权重(已含当日收益漂移)
    excluded = set()     # buffer模式下的当前排除名单
    ex_r, tw_l = [], []
    score_col = "score_sm" if mode == "smooth" else "score"

    for d, g in df.groupby("date", sort=True):
        g = g.dropna(subset=[weight_col, "ret_fwd_1d", score_col])
        if len(g) < 50:
            continue
        syms = g["symbol"].values
        wb = g[weight_col].values / g[weight_col].sum()
        sc = g[score_col].values
        can = g["tradable"].values == 1
        n = len(g)
        pct = sc.argsort().argsort() / n          # 分数百分位(0=最差)

        if mode == "buffer":
            enter, exit_ = frac, min(frac * buf_mult, 0.95)
            new_ex = set(syms[pct < enter])
            keep = {s for s, p in zip(syms, pct) if s in excluded and p < exit_}
            excluded = new_ex | keep
            drop = np.array([s in excluded for s in syms])
        else:
            drop = pct < frac

        drop = drop & can                          # 不可交易 → 躲不掉, 维持基准权重
        wp = wb * (~drop)
        wp = wp / wp.sum()

        # 换手: 与"上一日持仓经收益漂移后"的差, 非与上一日目标权重的差
        cur_prev = np.array([prev_w.get(s, 0.0) for s in syms])
        if cur_prev.sum() > 0:
            # 上一日持仓在昨日→今日间已按昨日收益漂移(prev_w存的已是漂移后)
            drift = cur_prev / cur_prev.sum()
        else:
            drift = np.zeros(n)
        extra = sum(v for s, v in prev_w.items() if s not in set(syms))  # 已出成分股
        tw_l.append(0.5 * (np.abs(wp - drift).sum() + extra) * 2)        # 单边换手

        r = np.nan_to_num(g["ret_fwd_1d"].values)
        ex_r.append(float(np.sum(wp * r) - np.sum(wb * r)))
        # 漂移到下一日
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
                       mdd=float((np.maximum.accumulate(np.cumsum(v)) - np.cumsum(v)).max()),
                       avg_bp=float(v.mean() * 1e4),
                       payoff=float(win.mean() / abs(loss.mean())) if len(loss) else np.nan,
                       pf=float(win.sum() / abs(loss.sum())) if len(loss) else np.nan)
    return res, te, tw.mean(), len(ex_r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True,
                    help="hs300/csi500/csi1000/gz2000")
    args = ap.parse_args()
    globals()["args"] = args    # build() 等模块级函数也要用

    df, dates = build()
    out = ["# 空头端规避 v2 — 换手控制下的剔除底部策略", "",
           f"> 验证段 {dates[0]}~{dates[-1]}, {len(dates)}天 | 信号={PRED} | "
           f"基准={C.INDEXES[args.index]['name']}指数权重 | **测试段未触碰**", "",
           "**对v1诊断的换手口径修正**: 基准权重每日随股价漂移, 持仓自动跟随、无需交易; "
           "换手改用 `|w_t − 上日持仓漂移后权重|`, 不再把基准自身漂移计入主动交易成本。", ""]

    for mode, title in [("plain", "A. 每日按当日分数剔除(无控制)"),
                        ("smooth", "B. 用过去5日平均分数剔除(信号平滑)"),
                        ("buffer", "C. 缓冲带(进入底部K%, 退出需回到底部2K%以上)")]:
        out += [f"## {title}", "",
                "| 剔除比例 | 超额毛 | 净10bp | 净20bp | 净30bp | IR(净20) | TE | 换手 | 交易日 | 胜率 | 单笔均bp | 盈亏比 | 盈利因子 | maxDD |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for frac in (0.05, 0.10, 0.20, 0.30):
            res, te, tw, nd = simulate(df, frac, mode)
            r20 = res[20]
            out.append(f"| 底部{frac*100:.0f}% | {res[0]['ann']*100:+.2f}% | "
                       f"{res[10]['ann']*100:+.2f}% | **{res[20]['ann']*100:+.2f}%** | "
                       f"{res[30]['ann']*100:+.2f}% | **{r20['ir']:.2f}** | {te*100:.2f}% | "
                       f"{tw:.3f} | {nd} | {r20['wr']*100:.0f}% | {r20['avg_bp']:+.2f} | "
                       f"{r20['payoff']:.2f} | {r20['pf']:.2f} | {r20['mdd']*100:.2f}% |")
        out.append("")

    txt = "\n".join(out) + "\n"
    fp = os.path.join(results_dir(args.index), "空头端规避_v2换手控制.md")
    with open(fp, "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
