# -*- coding: utf-8 -*-
"""重构"国证2000 代理域"(gz2000) — 替代中证2000, 规避其回溯重构风险.

背景:
  中证2000(932000.CSI) 2023年才发布, 2023年前成分是回溯重构的(成分数 2015年仅310只
  → 2023年才满2000), 存在域漂移与潜在事后信息。
  国证2000(399303.XSHE) 2014-03-28 起有真实历史, 但本机**只有指数价格、没有成分权重表**。

方案:
  按国证2000 的选样逻辑(剔除大中盘后按规模取2000只)自行重构域, 全部用 **T-1 流通市值**
  排名 → 天然 point-in-time, 无回溯偏差:
    每个交易日 t: 用 t-1 日流通市值对全A排名, 取第 RANK_LO..RANK_HI 名
    权重: 流通市值加权(与国证系列一致), 单票上限 CAP 防止个别票权重过大
  再用官方 399303.XSHE 指数收益对拍验证(相关性/年化差), 结果写入报告。

输出: {CACHE}/gz2000_weights.parquet  [date, index='gz2000', symbol, weight]
      (格式与 index_weights_all.parquet 一致, 由 build_weights 合并或单独加载)

用法: /usr/bin/python3 build_gz2000_universe.py [--workers 40]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

VAL_DIR = os.environ.get("IDXML_VALUATION","")
PRICE_DIR = os.environ.get("IDXML_PRICE_DAILY","")
IDX_PRICE_DIR = "${外部数据盘}"
GZ2000_CODE = "399303.XSHE"

RANK_LO, RANK_HI = 1001, 3000     # 剔除前1000大(≈国证1000), 取次2000只
CAP = 0.005                        # 单票权重上限 0.5%
OUT = os.path.join(C.CACHE, "gz2000_weights.parquet")


def _one(args):
    """用 prev_date 的流通市值定 date 当日的域与权重(T-1, 无前视)."""
    date, prev_date = args
    fv = os.path.join(VAL_DIR, f"{prev_date}.parquet")
    fp = os.path.join(PRICE_DIR, f"{prev_date}.parquet")
    if not (os.path.exists(fv) and os.path.exists(fp)):
        return None
    v = pd.read_parquet(fv, columns=["code", "circulating_market_cap"])
    p = pd.read_parquet(fp, columns=["code", "paused", "close"])
    m = v.merge(p, on="code", how="inner")
    m = m[(~m["paused"]) & m["circulating_market_cap"].notna() & (m["circulating_market_cap"] > 0)]
    m = m[m["code"].str.endswith((".XSHE", ".XSHG"))]
    m = m[~m["code"].str.startswith("688")]            # 剔科创板(国证2000 不含)
    m = m.sort_values("circulating_market_cap", ascending=False).reset_index(drop=True)
    sel = m.iloc[RANK_LO - 1:RANK_HI].copy()
    if len(sel) < 200:
        return None
    w = sel["circulating_market_cap"].values.astype("float64")
    w = w / w.sum()
    for _ in range(20):                                 # 迭代封顶再归一
        over = w > CAP
        if not over.any():
            break
        excess = (w[over] - CAP).sum()
        w[over] = CAP
        free = ~over
        if free.sum() == 0:
            break
        w[free] += excess * w[free] / w[free].sum()
    return pd.DataFrame({"date": date, "index": "gz2000",
                         "symbol": sel["code"].str[:6].values,
                         "weight": w.astype("float32")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=40)
    args = ap.parse_args()

    dates = C.trade_dates()
    jobs = [(dates[i], dates[i - 1]) for i in range(1, len(dates))]
    print(f"{len(jobs)} 个交易日, 域=流通市值排名 {RANK_LO}..{RANK_HI} (T-1定域, 单票封顶{CAP:.1%})",
          flush=True)
    parts = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_one, jobs, chunksize=16)):
            if r is not None:
                parts.append(r)
            if (i + 1) % 400 == 0:
                print(f"  {i+1}/{len(jobs)}", flush=True)
    w = pd.concat(parts, ignore_index=True)
    w.to_parquet(OUT, index=False)
    n = w.groupby("date")["symbol"].size()
    print(f"saved {OUT} {w.shape}; 日均成分 {n.mean():.0f} (min {n.min()}, max {n.max()})", flush=True)

    # ---- 与官方国证2000 对拍 ----
    print("\n=== 与官方国证2000(399303.XSHE) 对拍 ===", flush=True)
    px = []
    for d in dates:
        f = os.path.join(IDX_PRICE_DIR, f"{d}.parquet")
        if not os.path.exists(f):
            continue
        t = pd.read_parquet(f, columns=["date", "code", "close"])
        r = t[t["code"] == GZ2000_CODE]
        if len(r):
            px.append((d, float(r["close"].iloc[0])))
    idx = pd.DataFrame(px, columns=["date", "close"]).set_index("date")["close"]
    idx_ret = idx.pct_change()

    # 重构域的市值加权收益(用个股当日收益)
    rows = []
    for d, g in w.groupby("date"):
        f = os.path.join(PRICE_DIR, f"{d}.parquet")
        if not os.path.exists(f):
            continue
        p = pd.read_parquet(f, columns=["code", "close", "pre_close"])
        p["symbol"] = p["code"].str[:6]
        p["r"] = p["close"] / p["pre_close"] - 1
        m = g.merge(p[["symbol", "r"]], on="symbol", how="inner")
        if len(m) > 100:
            rows.append((d, float((m["weight"] * m["r"]).sum() / m["weight"].sum())))
    rec = pd.DataFrame(rows, columns=["date", "r"]).set_index("date")["r"]

    a = pd.concat([rec.rename("重构"), idx_ret.rename("官方")], axis=1).dropna()
    if len(a) > 100:
        corr = a["重构"].corr(a["官方"])
        print(f"  日收益相关 {corr:.4f}  ({len(a)} 天)")
        print(f"  年化: 重构 {a['重构'].mean()*242:+.2%} vs 官方 {a['官方'].mean()*242:+.2%} "
              f"(差 {(a['重构'].mean()-a['官方'].mean())*242:+.2%})")
        print(f"  年化跟踪误差 {(a['重构']-a['官方']).std()*np.sqrt(242):.2%}")
        for y, g in a.groupby(a.index.str[:4]):
            print(f"    {y}: 相关 {g['重构'].corr(g['官方']):.3f}  "
                  f"重构 {g['重构'].mean()*242:+.1%} / 官方 {g['官方'].mean()*242:+.1%}")
        a.to_parquet(os.path.join(C.CACHE, "gz2000_validation.parquet"))
    else:
        print("  官方指数数据不足, 无法对拍")


if __name__ == "__main__":
    main()
