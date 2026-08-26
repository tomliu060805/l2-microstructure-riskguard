# -*- coding: utf-8 -*-
"""合并四个指数的逐日成分权重 → 单一长表.

源: ${外部数据盘}{csi_300,csi_500,csi_1000,csi_2000}/<date>.parquet
    每文件列 [date, code(指数代码), stock_code(带.XSHE/.XSHG), weight]
出: ${IDXML_CACHE}/index_weights_all.parquet
    [date, index(hs300/csi500/csi1000/csi2000), symbol(6位), weight]

同时打印互斥性与成分数自检(四指数按市值分层, 理论上两两不相交)。

用法: /usr/bin/python3 build_weights.py [--workers 40]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def _one(args):
    key, path = args
    try:
        d = pd.read_parquet(path, columns=["date", "stock_code", "weight"])
    except Exception:
        return None
    d["date"] = d["date"].astype(str).str[:10]
    d["symbol"] = d["stock_code"].str[:6]
    d["index"] = key
    return d[["date", "index", "symbol", "weight"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=40)
    args = ap.parse_args()

    dates = set(C.trade_dates())
    jobs = []
    for key, meta in C.INDEXES.items():
        d = os.path.join(C.WEIGHT_ROOT, meta["dir"])
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".parquet") and fn[:-8] in dates:
                jobs.append((key, os.path.join(d, fn)))
    print(f"{len(jobs)} 个权重文件 ({len(C.INDEXES)} 指数)", flush=True)

    parts = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_one, jobs, chunksize=32)):
            if r is not None:
                parts.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(jobs)}", flush=True)
    w = pd.concat(parts, ignore_index=True)
    w["weight"] = w["weight"].astype("float32")
    os.makedirs(C.CACHE, exist_ok=True)
    w.to_parquet(C.WEIGHTS_ALL, index=False)
    print(f"saved {C.WEIGHTS_ALL} {w.shape}", flush=True)

    # 自检
    print("\n=== 各指数逐日成分数(中位) ===")
    for k in C.INDEXES:
        n = w[w["index"] == k].groupby("date")["symbol"].size()
        print(f"  {k:9s} {C.INDEXES[k]['name']:8s} 中位{n.median():.0f}只  "
              f"日期 {n.index.min()}~{n.index.max()}  {len(n)}天")
    print("\n=== 互斥性抽查(3个日期) ===")
    for d in sorted(w["date"].unique())[::max(1, len(w['date'].unique()) // 3)][:3]:
        g = w[w["date"] == d]
        sets = {k: set(x) for k, x in g.groupby("index")["symbol"]}
        ks = list(sets)
        ov = [(a, b, len(sets[a] & sets[b])) for i, a in enumerate(ks) for b in ks[i+1:]]
        uni = len(set().union(*sets.values()))
        bad = [x for x in ov if x[2] > 0]
        print(f"  {d}: 并集{uni}只, 两两交集 {'全为0 ✔' if not bad else bad}")


if __name__ == "__main__":
    main()
