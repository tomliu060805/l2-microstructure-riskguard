# -*- coding: utf-8 -*-
"""数据集加载与 walk-forward 调度(项目A/项目B 共用).

按指数加载: load_ds("csi1000")。所有模型共享同一数据入口与重训调度, 结果才可比。

从原 train_baselines.py 抽出, 保证所有模型(截面对照 / 下尾专用)共享
完全相同的数据入口与重训调度, 结果才可比。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

DS_ROOT = os.path.join(C.CACHE, "dataset")
PRED_ROOT = os.path.join(C.CACHE, "preds")
REFIT_EVERY = 63
MIN_TRAIN = 750


def ds_dir(index):
    """IDXML_DS_ROOT 可指向本地盘副本, 绕开共享数据盘 I/O 饱和。"""
    return os.path.join(os.environ.get("IDXML_DS_ROOT", DS_ROOT), index)


def pred_dir(index):
    d = os.path.join(PRED_ROOT, index)
    os.makedirs(d, exist_ok=True)
    return d


def load_ds(index):
    """按指数加载数据集(每个指数一张表, 秩变换已在该指数域内完成)。"""
    ds_dir_ = ds_dir(index)
    with open(os.path.join(ds_dir_, "feature_names.json")) as f:
        feats = json.load(f)
    ds = pd.read_parquet(os.path.join(ds_dir_, "ds.parquet"))
    ds["date"] = ds["date"].astype(str)
    ds = ds.sort_values(["date", "symbol"]).reset_index(drop=True)
    return ds, feats


def schedule(dates):
    """[(train_end_idx_exclusive, [predict dates])]

    每 REFIT_EVERY 个交易日重训(扩张窗); 训练集与预测块之间留 EMBARGO_DAYS,
    使训练集最晚样本的标签窗结束于预测块前一日 → 零重叠。
    """
    out = []
    i = MIN_TRAIN
    while i < len(dates):
        block = dates[i:i + REFIT_EVERY]
        if not block:
            break
        out.append((i - C.EMBARGO_DAYS, block))
        i += REFIT_EVERY
    return out


def split_train_es(didx, tr_mask, n_es=63):
    """训练窗内再切出末尾 n_es 天做早停, 与拟合段之间同样留 embargo。

    返回 (fit_idx, es_idx) 行号数组。
    """
    rows = np.where(tr_mask)[0]
    uniq = np.unique(didx[tr_mask])
    if len(uniq) <= 200:
        return rows, np.array([], np.int64)
    es_days = set(uniq[-n_es:])
    gap_days = set(uniq[-n_es - C.EMBARGO_DAYS:-n_es])
    drow = didx[rows]
    es_idx = rows[np.isin(drow, list(es_days))]
    fit_idx = rows[~np.isin(drow, list(es_days | gap_days))]
    return fit_idx, es_idx
