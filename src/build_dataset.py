# -*- coding: utf-8 -*-
"""组装训练数据集: 面板+标签 → **按指数**各自一张长表 parquet.

为什么按指数分开:
  四个域各自独立建模, 模型看到的截面就是该指数当日成分。若在并集(3800只)上做
  横截面秩, 中证2000 个股的秩会被沪深300大盘股拉偏, 与模型实际运行的截面不一致。
  因此**秩变换在各指数当日成分内计算**, 每个指数产出独立数据集。

- 特征: 每日**指数内**横截面秩变换 → [-0.5, 0.5], NaN 保留 (LGBM 原生处理; 线性基线填0=中位)
- 标签: 原始收益 ret_fwd_{1,2,5}d + 指数内横截面秩 y{1,5}d_rank (训练目标)
- 全窗口组装(含test段) — 组装≠评估, 评估侧由 common.assert_dev_dates 把关

输出: ${IDXML_CACHE}/dataset/<index>/ds.parquet
      [date, symbol, weight, tradable, ret_fwd_1d/2d/5d, y1d_rank, y5d_rank, <741特征>]
      ${IDXML_CACHE}/dataset/<index>/feature_names.json

用法: /usr/bin/python3 build_dataset.py --index csi1000 [--workers 30]
      /usr/bin/python3 build_dataset.py --index all          # 四个域依次构建
"""
import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

warnings.filterwarnings("ignore")

DS_ROOT = os.path.join(C.CACHE, "dataset")
_MEMBERS = None      # 子进程内惰性加载: {date -> DataFrame[symbol,weight]}


def rank_center(df):
    """逐列横截面 pct-rank → [-0.5,0.5], NaN 保留."""
    r = df.rank(pct=True, na_option="keep")
    return (r - 0.5).astype("float32")


def _init(index):
    global _MEMBERS
    _MEMBERS = C.index_weights_by_date(index)


def load_day(d, min_n=100):
    pp = os.path.join(C.PANEL_DIR, f"{d}.parquet")
    lp = os.path.join(C.LABEL_DIR, f"{d}.parquet")
    if not (os.path.exists(pp) and os.path.exists(lp)):
        return None
    mem = _MEMBERS.get(d)
    if mem is None or len(mem) < min_n:
        return None
    p = pd.read_parquet(pp)
    l = pd.read_parquet(lp)
    m = p.merge(l, on="symbol", how="inner").merge(mem, on="symbol", how="inner")
    if len(m) < min_n:
        return None
    feat_cols = [c for c in p.columns if c != "symbol"]
    m[feat_cols] = rank_center(m[feat_cols])           # 指数内截面秩
    m["y1d_rank"] = rank_center(m[["ret_fwd_1d"]])["ret_fwd_1d"]
    m["y5d_rank"] = rank_center(m[["ret_fwd_5d"]])["ret_fwd_5d"]
    m.insert(0, "date", d)
    return m


def build_index(index, workers):
    out_dir = os.path.join(DS_ROOT, index)
    os.makedirs(out_dir, exist_ok=True)
    dates = C.trade_dates()
    parts = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(index,)) as ex:
        for i, r in enumerate(ex.map(load_day, dates, chunksize=8)):
            if r is not None:
                parts.append(r)
            if (i + 1) % 400 == 0:
                print(f"  [{index}] {i+1}/{len(dates)}", flush=True)
    if not parts:
        print(f"  [{index}] 无数据, 跳过", flush=True)
        return
    ds = pd.concat(parts, ignore_index=True)
    ds["date"] = ds["date"].astype("category")
    ds["symbol"] = ds["symbol"].astype("category")
    feat_cols = [c for c in ds.columns if "|" in c or c == "coverage"]
    with open(os.path.join(out_dir, "feature_names.json"), "w") as f:
        json.dump(feat_cols, f)
    out = os.path.join(out_dir, "ds.parquet")
    ds.to_parquet(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)
    n_per_day = ds.groupby("date", observed=True)["symbol"].size()
    print(f"  [{index}] rows {len(ds)} days {ds.date.nunique()} 特征 {len(feat_cols)} "
          f"日均成分 {n_per_day.mean():.0f} → {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True,
                    help="hs300 / csi500 / csi1000 / csi2000 / all")
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_args()
    idxs = C.ALL_INDEXES if args.index == "all" else [args.index]
    for ix in idxs:
        assert ix in C.INDEXES, f"未知指数 {ix}"
        print(f"=== 构建 {ix} ({C.INDEXES[ix]['name']})", flush=True)
        build_index(ix, args.workers)


if __name__ == "__main__":
    main()
