# -*- coding: utf-8 -*-
"""下尾专用模型: 不对称目标, 专攻"谁会跌"而非"谁会涨".

对照组 = 项目1的对称回归模型(lgbm_y5d_rank, 预测收益秩)。
本脚本三种不对称建模:
  clf     二分类 — 标签 = 是否落入未来5日收益的最差20%(binary logloss)
  clf_w   二分类 + 样本加权 — 越靠近极端下尾权重越高(强调"深跌"而非"一般弱")
  quant   分位数回归 — 只拟合收益分布的10%分位(LightGBM objective=quantile, alpha=0.1)

三者共用与项目1完全相同的 walk-forward 调度/embargo/特征集, 差别只在目标函数,
增量才可归因于"不对称目标"本身。

用法: IDXML_DS_ROOT=../../data /usr/bin/python3 train_tail.py --obj clf --seeds 3 --threads 60
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
RESULTS = os.path.join(HERE, "..", "results")
import common as C
from dsutil import load_ds, schedule, split_train_es, pred_dir
from btutil import q_thresh

warnings.filterwarnings("ignore")

# ⚠️ y5d_rank 是中心化秩 ∈[-0.5,+0.5](见 btutil.q_thresh), 阈值必须转换
TAIL_Q = 0.20      # "下尾" 定义: 未来5日收益横截面最差20%
DEEP_Q = 0.05      # 深跌: 最差5%(加权版给更高权重)
TAIL_T = q_thresh(TAIL_Q)   # = -0.30
DEEP_T = q_thresh(DEEP_Q)   # = -0.45


def fit_predict(obj, Xtr, ytr_rank, Xte, didx_tr, tr_mask, seeds, threads):
    """ytr_rank: 训练样本的横截面收益秩(0=最差, 1=最好)."""
    import lightgbm as lgb
    fit_idx_rel, es_idx_rel = split_train_es(didx_tr, np.ones(len(ytr_rank), bool))
    preds = []
    for seed in seeds:
        if obj == "quant":
            y = ytr_rank
            params = dict(objective="quantile", alpha=0.10, metric="quantile")
        else:
            y = (ytr_rank <= TAIL_T).astype(np.float32)   # 1 = 落入下尾
            params = dict(objective="binary", metric="binary_logloss")
        w = None
        if obj == "clf_w":
            # 深跌样本权重更高: 最差5% 给 3.0, 5~20% 给 1.5, 其余 1.0
            w = np.ones(len(ytr_rank), np.float32)
            w[ytr_rank <= TAIL_T] = 1.5
            w[ytr_rank <= DEEP_T] = 3.0
        m = lgb.LGBMRegressor if obj == "quant" else lgb.LGBMClassifier
        model = m(n_estimators=1000, learning_rate=0.05, num_leaves=63,
                  min_child_samples=200, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.6, reg_lambda=1.0, random_state=seed,
                  n_jobs=threads, verbose=-1, **params)
        kw = {}
        if len(es_idx_rel):
            kw = dict(eval_set=[(Xtr[es_idx_rel], y[es_idx_rel])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            if w is not None:
                kw["eval_sample_weight"] = [w[es_idx_rel]]
        fw = w[fit_idx_rel] if w is not None else None
        model.fit(Xtr[fit_idx_rel], y[fit_idx_rel], sample_weight=fw, **kw)
        if obj == "quant":
            preds.append(model.predict(Xte))
        else:
            # 分类器输出"属于下尾的概率" → 取负号使方向与收益一致(高分=好)
            preds.append(-model.predict_proba(Xte)[:, 1])
    return np.mean(preds, 0)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--index", required=True,

                    help="hs300/csi500/csi1000/csi2000 — 每个指数域独立建模")
    ap.add_argument("--obj", choices=["clf", "clf_w", "quant"], required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=60)
    args = ap.parse_args()

    ds, feats = load_ds(args.index)
    all_dates = sorted(ds["date"].unique())
    dev_dates = [d for d in all_dates if not C.is_test_date(d)]
    C.assert_dev_dates(dev_dates, "train_tail")
    ds = ds[ds["date"].isin(dev_dates)].reset_index(drop=True)

    date_to_idx = {d: i for i, d in enumerate(dev_dates)}
    didx = ds["date"].map(date_to_idx).values.astype(np.int32)
    X = ds[feats].values.astype(np.float32)
    y = ds["y5d_rank"].values.astype(np.float32)   # 中心化横截面秩 ∈[-0.5,+0.5]
    valid = ~np.isnan(y)

    sched = schedule(dev_dates)
    print(f"tail-{args.obj} rows={len(ds)} feats={len(feats)} refits={len(sched)}", flush=True)

    out = []
    for bi, (tr_end, block) in enumerate(sched):
        t0 = time.time()
        tr_mask = (didx < tr_end) & valid
        te_mask = np.isin(didx, [date_to_idx[d] for d in block])
        if tr_mask.sum() < 1000 or te_mask.sum() == 0:
            continue
        s = fit_predict(args.obj, X[tr_mask], y[tr_mask], X[te_mask],
                        didx[tr_mask], tr_mask, list(range(args.seeds)), args.threads)
        blk = ds.loc[te_mask, ["date", "symbol"]].copy()
        blk["score"] = s.astype(np.float32)
        out.append(blk)
        print(f"block {bi+1}/{len(sched)} {block[0]}..{block[-1]} "
              f"train={tr_mask.sum()} {time.time()-t0:.0f}s", flush=True)

    preds = pd.concat(out, ignore_index=True)
    fp = os.path.join(pred_dir(args.index), f"tail_{args.obj}_y5d.parquet")
    preds.to_parquet(fp, index=False)
    print("saved", fp, len(preds), flush=True)


if __name__ == "__main__":
    main()
