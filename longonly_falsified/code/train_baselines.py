# -*- coding: utf-8 -*-
"""Walk-forward 训练: IC加权线性基线 + LightGBM. 共享同一调度, 输出统一格式预测.

调度: 每 REFIT_EVERY=63 个交易日 refit 一次, 扩张训练窗口;
     训练集截止 = 预测块首日 index − 1 − EMBARGO(5天, = 最长标签期限) → purged.
目标: y5d_rank (5日前向收益横截面秩, 主) / y1d_rank (对照).
LGBM: 3-5 种子, 内部早停(训练窗末63天做es验证, 与训练主体间再留5天embargo).

默认只产出 dev 段(< TEST_START)预测; test 段调用被 common.assert_dev_dates 拦截,
最终评估需 IDXML_TEST_UNLOCK=1.

用法:
  /usr/bin/python3 train_baselines.py --model linear --target y5d_rank
  /usr/bin/python3 train_baselines.py --model lgbm --target y5d_rank --seeds 3
"""
import argparse
import json
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

warnings.filterwarnings("ignore")

from dsutil import load_ds, schedule, ds_dir, pred_dir, REFIT_EVERY, MIN_TRAIN

TOPK_LINEAR = 60


def fit_predict_linear(Xtr, ytr, Xte):
    X = np.nan_to_num(Xtr, nan=0.0)
    y = ytr
    Xc = X - X.mean(0)
    yc = y - y.mean()
    denom = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12
    ic = (Xc * yc[:, None]).sum(0) / denom  # 特征/目标均为秩 → 即Spearman IC
    k = np.argsort(-np.abs(ic))[:TOPK_LINEAR]
    w = ic[k]
    return np.nan_to_num(Xte, nan=0.0)[:, k] @ w


def fit_predict_lgbm(Xtr, ytr, Xte, tr_dates_idx, seeds, threads, imp_out=None,
                     sample_weight=None):
    import lightgbm as lgb
    # 内部早停切分: 训练窗末63天为es验证, 之间再留5天embargo
    uniq = np.unique(tr_dates_idx)
    if len(uniq) > 200:
        es_days = set(uniq[-63:])
        gap_days = set(uniq[-63 - C.EMBARGO_DAYS:-63])
        es_mask = np.isin(tr_dates_idx, list(es_days))
        fit_mask = ~np.isin(tr_dates_idx, list(es_days | gap_days))
    else:
        es_mask = np.zeros(len(ytr), bool)
        fit_mask = ~es_mask
    preds = []
    for seed in seeds:
        m = lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, num_leaves=63,
            min_child_samples=200, feature_fraction=0.6, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=1.0, random_state=seed, n_jobs=threads,
            verbose=-1)
        kw = {}
        if es_mask.any():
            kw = dict(eval_set=[(Xtr[es_mask], ytr[es_mask])],
                      callbacks=[__import__("lightgbm").early_stopping(50, verbose=False)])
            if sample_weight is not None:
                kw["eval_sample_weight"] = [sample_weight[es_mask]]
        if sample_weight is not None:
            kw["sample_weight"] = sample_weight[fit_mask]
        m.fit(Xtr[fit_mask], ytr[fit_mask], **kw)
        preds.append(m.predict(Xte))
        if imp_out is not None:
            imp_out.append(m.booster_.feature_importance(importance_type="gain"))
    return np.mean(preds, 0)


def fit_predict_xgb(Xtr, ytr, Xte, tr_dates_idx, seeds, threads):
    import xgboost as xgb
    uniq = np.unique(tr_dates_idx)
    if len(uniq) > 200:
        es_days = set(uniq[-63:])
        gap_days = set(uniq[-63 - C.EMBARGO_DAYS:-63])
        es_mask = np.isin(tr_dates_idx, list(es_days))
        fit_mask = ~np.isin(tr_dates_idx, list(es_days | gap_days))
    else:
        es_mask = np.zeros(len(ytr), bool)
        fit_mask = ~es_mask
    preds = []
    for seed in seeds:
        m = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            min_child_weight=200, subsample=0.8, colsample_bytree=0.6,
            reg_lambda=1.0, random_state=seed, n_jobs=threads,
            tree_method="hist", early_stopping_rounds=50 if es_mask.any() else None,
            verbosity=0)
        kw = dict(eval_set=[(Xtr[es_mask], ytr[es_mask])], verbose=False) if es_mask.any() else {}
        m.fit(Xtr[fit_mask], ytr[fit_mask], **kw)
        preds.append(m.predict(Xte))
    return np.mean(preds, 0)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--index", required=True,

                    help="hs300/csi500/csi1000/csi2000 — 每个指数域独立建模")
    ap.add_argument("--model", choices=["linear", "lgbm", "xgb"], required=True)
    ap.add_argument("--target", default="y5d_rank")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--tag", default="")
    ap.add_argument("--feat-filter", default=None,
                    help="正则: 只保留匹配的特征列(消融用), 如 '\\|mean$'")
    ap.add_argument("--feat-list", default=None,
                    help="文件: 每行一个特征名, 只用这些列(top-K消融用)")
    ap.add_argument("--dump-importance", action="store_true")
    ap.add_argument("--train-start", default=None,
                    help="训练窗起点(YYYY-MM-DD), 用于训练窗敏感性对照。"
                         "只截断训练数据, 预测块(验证段)不变 → 结果可直接与全历史版比较")
    args = ap.parse_args()

    PRED_DIR = pred_dir(args.index)
    ds, feats = load_ds(args.index)
    if args.feat_filter:
        import re
        feats = [f for f in feats if re.search(args.feat_filter, f)]
        print(f"feat-filter '{args.feat_filter}' -> {len(feats)} features", flush=True)
    if args.feat_list:
        keep = set(open(args.feat_list).read().split())
        feats = [f for f in feats if f in keep]
        print(f"feat-list -> {len(feats)} features", flush=True)
    all_dates = sorted(ds["date"].unique())
    if os.environ.get("IDXML_TEST_UNLOCK") != "1":
        dev_dates = [d for d in all_dates if not C.is_test_date(d)]
    else:
        dev_dates = all_dates
    C.assert_dev_dates(dev_dates, "train_baselines") if os.environ.get("IDXML_TEST_UNLOCK") != "1" else None
    ds = ds[ds["date"].isin(dev_dates)].reset_index(drop=True)
    if args.train_start:
        # 只砍训练样本, 预测块仍覆盖完整验证段 → 与全历史版可比
        n0 = len(ds)
        ds = ds[(ds["date"] >= args.train_start) | (ds["date"] >= C.VAL_START)].reset_index(drop=True)
        dev_dates = [d for d in dev_dates if d >= args.train_start or d >= C.VAL_START]
        print(f"train-start={args.train_start}: 样本 {n0} → {len(ds)}, 日期 {len(dev_dates)} 天", flush=True)

    date_to_idx = {d: i for i, d in enumerate(dev_dates)}
    didx = ds["date"].map(date_to_idx).values.astype(np.int32)
    X = ds[feats].values.astype(np.float32)
    y = ds[args.target].values.astype(np.float32)
    valid = ~np.isnan(y)

    sched = schedule(dev_dates)
    print(f"{args.model} target={args.target} rows={len(ds)} refits={len(sched)}", flush=True)

    out = []
    importances = {}
    for bi, (tr_end, block) in enumerate(sched):
        t0 = time.time()
        tr_mask = (didx < tr_end) & valid
        te_mask = np.isin(didx, [date_to_idx[d] for d in block])
        if tr_mask.sum() < 1000 or te_mask.sum() == 0:
            continue
        if args.model == "linear":
            s = fit_predict_linear(X[tr_mask], y[tr_mask], X[te_mask])
        elif args.model == "xgb":
            s = fit_predict_xgb(X[tr_mask], y[tr_mask], X[te_mask],
                                didx[tr_mask], list(range(args.seeds)), args.threads)
        else:
            imp = [] if args.dump_importance else None
            s = fit_predict_lgbm(X[tr_mask], y[tr_mask], X[te_mask],
                                 didx[tr_mask], list(range(args.seeds)), args.threads,
                                 imp_out=imp)
            if imp:
                importances[block[0]] = np.mean(imp, 0)
        blk = ds.loc[te_mask, ["date", "symbol"]].copy()
        blk["score"] = s.astype(np.float32)
        out.append(blk)
        print(f"block {bi+1}/{len(sched)} {block[0]}..{block[-1]} "
              f"train={tr_mask.sum()} {time.time()-t0:.0f}s", flush=True)

    preds = pd.concat(out, ignore_index=True)
    name = f"{args.model}_{args.target}{('_'+args.tag) if args.tag else ''}"
    if os.environ.get("IDXML_TEST_UNLOCK") == "1":
        name += "_WITHTEST"
    fp = os.path.join(PRED_DIR, f"{name}.parquet")
    preds.to_parquet(fp, index=False)
    print("saved", fp, len(preds), flush=True)
    if importances:
        impdf = pd.DataFrame(importances, index=feats)
        impfp = os.path.join(PRED_DIR, f"importance_{name}.parquet")
        impdf.to_parquet(impfp)
        print("saved", impfp, flush=True)


if __name__ == "__main__":
    main()
