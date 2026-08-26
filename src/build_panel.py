# -*- coding: utf-8 -*-
"""日频特征面板构建: 247个L2 5min因子 → 每股每日聚合特征.

域: **全市场**(因子库覆盖的全部股票, 约5100只), 各项目按 index 取子集。
   为什么不按指数并集建: 重构的 gz2000 域(市值秩1001-3000)只有 64-87% 落在 CSI 四指数
   并集内, 按并集建会缺 267-712 只/日。全市场建一次, 以后换任何域都不必重建。

无前视设计:
  - 只用当日 09:30..14:55 共48根bar(剔除15:00那根), 信号在收盘前5分钟已知,
    可按收盘竞价成交; 标签是 15:00 起算的前向收益 → 特征严格先于标签.
  - 聚合仅在日内, 不做任何跨日滚动(跨日特征在训练脚本里由已落盘面板 shift 生成).

每因子3个聚合: mean(全日均值) / std(日内波动) / lh(尾盘14:00-14:55均值)
输出: ${IDXML_CACHE}/panel_daily/<date>.parquet
  [symbol, coverage, <factor>|mean, <factor>|std, <factor>|lh, ...]  (~741特征列, float32)

用法: /usr/bin/python3 build_panel.py [--workers 60] [--dates ...]
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

warnings.filterwarnings("ignore")

_T1455 = pd.Timestamp("2000-01-01 14:55:00").time()
_T1400 = pd.Timestamp("2000-01-01 14:00:00").time()


def build_one(date, members, facs):
    out_path = os.path.join(C.PANEL_DIR, f"{date}.parquet")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        return "skip"
    pieces = []
    nvalid_total = None
    for fac in facs:
        fp = os.path.join(C.FACTOR_ROOT, fac, f"{date}.parquet")
        if not os.path.exists(fp):
            continue
        try:
            d = pd.read_parquet(fp)
        except Exception:
            return f"badfile:{fac}"
        if members is not None:
            d = d[d["symbol"].isin(members)]
        t = d["datetime"].dt.time
        d = d[t <= _T1455]  # 剔除15:00 bar → 无前视
        v = d[fac].astype("float64")
        g = d.assign(_v=v).groupby("symbol")["_v"]
        agg = pd.DataFrame({
            f"{fac}|mean": g.mean(),
            f"{fac}|std": g.std(),
        })
        lh = d[d["datetime"].dt.time >= _T1400]
        agg[f"{fac}|lh"] = lh.groupby("symbol")[fac].mean()
        nv = g.count()
        nvalid_total = nv if nvalid_total is None else nvalid_total.add(nv, fill_value=0)
        pieces.append(agg.astype("float32"))
    if not pieces:
        return "empty"
    out = pd.concat(pieces, axis=1)
    out.insert(0, "coverage", (nvalid_total / (len(pieces) * 48.0)).astype("float32"))
    out = out.reset_index().rename(columns={"index": "symbol"})
    tmp = out_path + ".tmp"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--dates", nargs="*", default=None)
    ap.add_argument("--universe", default="full",
                    help="full=全市场(默认,推荐) | index=四指数并集")
    args = ap.parse_args()

    dates = args.dates or C.trade_dates()
    mem = None if args.universe == "full" else C.members_by_date()
    facs = C.factor_names()
    os.makedirs(C.PANEL_DIR, exist_ok=True)
    print(f"{len(dates)} dates x {len(facs)} factors", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    stats = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_one, d, (mem[d] if mem else None), facs): d
                for d in dates if (mem is None or mem.get(d))}
        for i, fu in enumerate(as_completed(futs)):
            r = fu.result()
            key = r.split(":")[0]
            stats[key] = stats.get(key, 0) + 1
            if (i + 1) % 100 == 0:
                print(f"{i+1}/{len(futs)} {stats}", flush=True)
    print("done", stats, flush=True)


if __name__ == "__main__":
    main()
