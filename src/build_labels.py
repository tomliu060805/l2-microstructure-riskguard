# -*- coding: utf-8 -*-
"""标签构建: 每日四指数并集成分股的前向收益 + 收盘可交易性.

标签定义 (已核验: ret_T1d@15:00 = 次日收盘/今收 - 1, 复权口径, corr=0.9994 vs 次日1min累乘):
  ret_fwd_{1d,2d,5d} = fwd_ret_with_open 当日 15:00 bar 的 ret_T{1,2,5}d
可交易性: limit_status(**14:55**) != 0 或 paused != 0 → tradable=0(14:55 才是决策时点)
  (收盘涨跌停/停牌无法按收盘价成交, 回测时买卖两侧都要过滤)

输出: ${IDXML_CACHE}/labels/<date>.parquet
  [symbol, ret_fwd_1d, ret_fwd_2d, ret_fwd_5d, tradable]

用法: /usr/bin/python3 build_labels.py [--workers 40] [--dates d1 d2 ...]
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


def build_one(date, members):
    out_path = os.path.join(C.LABEL_DIR, f"{date}.parquet")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return "skip"
    fp = os.path.join(C.FWDRET_ROOT, f"{date}.parquet")
    if not os.path.exists(fp):
        return "no_fwdret"
    f = pd.read_parquet(fp, columns=["code", "datetime", "ret_T1d", "ret_T2d", "ret_T5d"])
    f = f[f["datetime"].dt.time == C._BAR_1500].copy()
    f["symbol"] = f["code"].str[:6]
    f = f[f["symbol"].isin(members)]
    f = f.drop_duplicates("symbol").set_index("symbol")

    lp = os.path.join(C.LIMIT_ROOT, f"{date}.parquet")
    if os.path.exists(lp):
        lim = pd.read_parquet(lp)
        # 用14:55状态(决策时点已知), 非15:00(与14:55不一致率仅0.317%, 但14:55才是无窥视口径)
        lim = lim[lim["datetime"].dt.time == pd.Timestamp("2000-01-01 14:55:00").time()]
        lim = lim.drop_duplicates("symbol").set_index("symbol")[["limit_status", "paused"]]
        f = f.join(lim, how="left")
    else:
        f["limit_status"] = np.nan
        f["paused"] = np.nan

    out = pd.DataFrame(index=f.index)
    out["ret_fwd_1d"] = f["ret_T1d"].astype("float32")
    out["ret_fwd_2d"] = f["ret_T2d"].astype("float32")
    out["ret_fwd_5d"] = f["ret_T5d"].astype("float32")
    # limit/paused 缺失按可交易处理(缺失≈数据缺口, 不作系统性剔除)
    out["tradable"] = (
        (f["limit_status"].fillna(0) == 0) & (f["paused"].fillna(0) == 0)
    ).astype("int8")
    out = out.reset_index()
    tmp = out_path + ".tmp"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--dates", nargs="*", default=None)
    ap.add_argument("--allow-test", action="store_true",
                    help="标签构建覆盖全窗口(含test段)是允许的: 构建≠评估")
    args = ap.parse_args()

    dates = args.dates or C.trade_dates()
    mem = C.members_by_date()
    os.makedirs(C.LABEL_DIR, exist_ok=True)

    from concurrent.futures import ProcessPoolExecutor
    stats = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_one, d, mem[d]): d for d in dates if mem.get(d)}
        for i, fu in enumerate(futs):
            pass
        from concurrent.futures import as_completed
        for i, fu in enumerate(as_completed(futs)):
            r = fu.result()
            stats[r] = stats.get(r, 0) + 1
            if (i + 1) % 200 == 0:
                print(f"{i+1}/{len(futs)} {stats}", flush=True)
    print("done", stats, flush=True)


if __name__ == "__main__":
    main()
