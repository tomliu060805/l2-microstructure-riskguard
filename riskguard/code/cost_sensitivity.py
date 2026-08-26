# -*- coding: utf-8 -*-
"""A2 成本敏感性分析 — 20bp 假设在小盘域站得住吗?

## 动机
跨域最亮眼的结果(中证2000 IR 3.08 / 中性化后仍 IR 2.02)建立在 **20bp 双边成本**
假设上。但中证2000 中位流通市值只有 32 亿, 微盘股价差更宽、盘口更薄、冲击更大。
若真实成本是 2~3 倍, "往小盘走"这个方向可能根本不成立。

## 三层做法
  L1 成本网格   0~120bp 细网格 × 三域 × 中性化前后 → 盈亏平衡点
  L2 真实价差   从 L1 逐笔五档快照实测各域相对价差(半价差=最小可实现成本)
  L3 冲击成本   Amihud 式: 用日成交额 + 组合规模估冲击, 得"规模→成本→超额"链条

用法: /usr/bin/python3 cost_sensitivity.py [--spread-days 12]
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
from style_neutralize import build_styles, neutralize, NAME, PRED_DEFAULT

warnings.filterwarnings("ignore")

TICK_ROOT = "${外部数据盘}1"
DAILY_ROOT = os.environ.get("IDXML_PRICE_DAILY","")


def spread_one_day(args):
    """某日各股相对价差(时间加权近似: 用所有快照的中位数)与盘口深度."""
    date, members = args
    fp = os.path.join(TICK_ROOT, f"{date}.parquet")
    if not os.path.exists(fp):
        return None
    try:
        d = pd.read_parquet(fp, columns=["code", "datetime", "a1_p", "b1_p", "a1_v", "b1_v"])
    except Exception:
        return None
    d["symbol"] = d["code"].str[:6]
    d = d[d["symbol"].isin(members)]
    # 只取连续竞价时段的有效报价
    t = d["datetime"].dt.time
    d = d[(t >= pd.Timestamp("2000-01-01 09:30").time()) &
          (t <= pd.Timestamp("2000-01-01 14:57").time())]
    d = d[(d["a1_p"] > 0) & (d["b1_p"] > 0)]
    if d.empty:
        return None
    mid = (d["a1_p"] + d["b1_p"]) / 2
    d["rel_spread"] = (d["a1_p"] - d["b1_p"]) / mid          # 相对价差
    d["touch_amt"] = (d["a1_v"] + d["b1_v"]) / 2 * mid       # 一档挂单金额
    g = d.groupby("symbol").agg(rel_spread=("rel_spread", "median"),
                                touch_amt=("touch_amt", "median"))
    g["date"] = date
    return g.reset_index()


def load_daily(date):
    fp = os.path.join(DAILY_ROOT, f"{date}.parquet")
    if not os.path.exists(fp):
        return None
    d = pd.read_parquet(fp, columns=["date", "code", "money"])
    d["symbol"] = d["code"].str[:6]
    d["date"] = d["date"].astype(str)
    return d[["date", "symbol", "money"]]


def measure_spread(univ, dates, n_days):
    """抽样若干天实测该域的相对价差与一档深度."""
    mem = C.members_by_date([univ])
    step = max(len(dates) // n_days, 1)
    sample = dates[::step][:n_days]
    args = [(d, mem.get(d) or set()) for d in sample]
    with ProcessPoolExecutor(min(n_days, 12)) as ex:
        parts = [r for r in ex.map(spread_one_day, args) if r is not None]
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    return dict(
        n_days=len(parts),
        rel_spread_bp=float(df["rel_spread"].median() * 1e4),
        half_spread_bp=float(df["rel_spread"].median() * 1e4 / 2),
        touch_amt_wan=float(df["touch_amt"].median() / 1e4),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spread-days", type=int, default=12)
    ap.add_argument("--index", nargs="*", default=C.ALL_INDEXES)
    args = ap.parse_args()

    grid = list(range(0, 121, 10))
    res_all, spread_all, adv_all = {}, {}, {}

    for univ in args.index:
        if not os.path.exists(os.path.join(ds_dir(univ), "ds.parquet")):
            continue
        print(f"=== {univ}", flush=True)
        df, dates = build(univ, pred_name=PRED_DEFAULT)
        st = build_styles(ds_dir(univ), dates)
        df = df.merge(st, on=["date", "symbol"], how="left")
        df["score_neu"], _ = neutralize(df)

        for tag, col in [("原始", "score"), ("中性化", "score_neu")]:
            d2 = df.sort_values(["symbol", "date"]).copy()
            d2["score_sm"] = d2.groupby("symbol")[col].transform(
                lambda s: s.rolling(5, min_periods=1).mean())
            d2 = d2.sort_values(["date", "symbol"]).reset_index(drop=True)
            res, te, tw, nd = simulate(d2, 0.30, "smooth", cost_bps=tuple(grid))
            res_all[(univ, tag)] = dict(res=res, te=te, tw=tw,
                                        be=res[0]["ann"] / (tw * ANN / 1e4))
            print(f"  {tag}: 毛{res[0]['ann']*100:+.2f}% 换手{tw:.3f} "
                  f"打平{res_all[(univ,tag)]['be']:.0f}bp", flush=True)

        sp = measure_spread(univ, dates, args.spread_days)
        spread_all[univ] = sp
        if sp:
            print(f"  实测相对价差 {sp['rel_spread_bp']:.1f}bp "
                  f"(半价差{sp['half_spread_bp']:.1f}bp) 一档深度{sp['touch_amt_wan']:.1f}万",
                  flush=True)
        # 日均成交额(用于冲击估计)
        step = max(len(dates) // args.spread_days, 1)
        with ProcessPoolExecutor(12) as ex:
            dl = [r for r in ex.map(load_daily, dates[::step][:args.spread_days]) if r is not None]
        if dl:
            dl = pd.concat(dl, ignore_index=True)
            mem = C.members_by_date([univ])
            keep = dl.apply(lambda r: r["symbol"] in (mem.get(r["date"]) or set()), axis=1)
            adv_all[univ] = float(dl[keep]["money"].median() / 1e8)
            print(f"  成分股日均成交额中位 {adv_all[univ]:.2f} 亿", flush=True)

    # ---------- 报告 ----------
    out = ["# A2 成本敏感性分析 — 20bp 假设在小盘域站得住吗?", "",
           "> 验证段 2022-07-01~2024-06-28 | 信号=下尾专用模型 | 落地形态=5日平滑+剔除底部30%",
           "> | **测试段未触碰**", "",
           "## L1. 成本网格与盈亏平衡点", "",
           "| 域 | 口径 | 超额毛 | 换手 | 净20bp | 净40bp | 净60bp | 净80bp | **打平成本** |",
           "|---|---|---|---|---|---|---|---|---|"]
    for (u, tag), v in res_all.items():
        r = v["res"]
        out.append(f"| {NAME(u)} | {tag} | {r[0]['ann']*100:+.2f}% | {v['tw']:.3f} | "
                   f"{r[20]['ann']*100:+.2f}% | {r[40]['ann']*100:+.2f}% | "
                   f"{r[60]['ann']*100:+.2f}% | {r[80]['ann']*100:+.2f}% | "
                   f"**{v['be']:.0f}bp** |")

    out += ["", "## L2. 实测盘口价差(L1 逐笔五档快照, 连续竞价时段中位数)", "",
            "| 域 | 抽样天数 | 相对价差 | **半价差(最小可实现成本)** | 一档挂单金额 | 日均成交额中位 |",
            "|---|---|---|---|---|---|"]
    for u, sp in spread_all.items():
        if not sp:
            continue
        adv = adv_all.get(u)
        out.append(f"| {NAME(u)} | {sp['n_days']} | {sp['rel_spread_bp']:.1f}bp | "
                   f"**{sp['half_spread_bp']:.1f}bp** | {sp['touch_amt_wan']:.1f}万 | "
                   f"{adv:.2f}亿 |" if adv else "|")

    out += ["", "半价差是**理想情况下的单边成本下界**(以中价成交需付半价差), ",
            "实盘还要加冲击成本、机会成本与佣金印花税(A股卖出印花税 5bp 单边)。", ""]

    txt = "\n".join(out) + "\n"
    with open(os.path.join(RESULTS, "A2_成本敏感性_四域.md"), "w") as f:
        f.write(txt)
    import json
    with open(os.path.join(RESULTS, "A2_成本敏感性.json"), "w") as f:
        json.dump({"spread": spread_all, "adv_yi": adv_all,
                   "be": {f"{u}/{t}": v["be"] for (u, t), v in res_all.items()}},
                  f, ensure_ascii=False, indent=2)
    print(txt)


if __name__ == "__main__":
    main()
