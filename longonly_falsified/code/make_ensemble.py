# -*- coding: utf-8 -*-
"""LGBM + XGB 秩集成: 逐日把各模型分数转横截面秩后等权平均.

用法: /usr/bin/python3 make_ensemble.py --preds lgbm_y5d_rank xgb_y5d_rank --out ens_lgbm_xgb
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
RESULTS = os.path.join(HERE, "..", "results")
import common as C
from dsutil import pred_dir




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--index", default="csi500")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    acc = None
    for p in args.preds:
        d = pd.read_parquet(os.path.join(pred_dir(args.index), f"{p}.parquet"))
        d["r"] = d.groupby("date")["score"].rank(pct=True)
        d = d[["date", "symbol", "r"]].rename(columns={"r": p})
        acc = d if acc is None else acc.merge(d, on=["date", "symbol"], how="inner")
    acc["score"] = acc[args.preds].mean(axis=1).astype("float32")
    out = acc[["date", "symbol", "score"]]
    fp = os.path.join(pred_dir(args.index), f"{args.out}.parquet")
    out.to_parquet(fp, index=False)
    print("saved", fp, len(out), "| 成分相关(秩):", flush=True)
    print(acc[args.preds].corr(method="spearman").round(3).to_string())


if __name__ == "__main__":
    main()
