# -*- coding: utf-8 -*-
"""NN 对照组: 与 LGBM 完全相同的 walk-forward 调度/embargo/目标.

两个架构:
  mlp: 当日741特征(与LGBM同信息集) → 512-128-1, 检验"同信息下 NN vs GBDT"
  gru: 过去L=5个交易日 × 247个|mean特征的序列 → GRU(96) → head,
       检验"因子时序动态是否有增量"(序列只取当前行之前(含当日)的同股历史行, 无前视;
       成分缺席日按最近可得观测回看, 属既知历史信息)
目标: y5d_rank, MSE. 早停: 训练窗末63天(embargo 5天), patience 3, max 20 epochs.

用法: /usr/bin/python3 train_nn.py --arch mlp --seeds 2 --threads 40
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
from dsutil import load_ds, schedule, pred_dir

warnings.filterwarnings("ignore")

SEQ_L = 5


def build_seq_index(ds):
    """每行 → 同股过去L行(含当日)的行号, 不足向前重复首行."""
    n = len(ds)
    idx = np.arange(n)
    order = ds.sort_values(["symbol", "date"]).index.values
    seq = np.empty((n, SEQ_L), np.int64)
    sym = ds["symbol"].values
    pos_in_group = np.empty(n, np.int64)
    group_start = {}
    prev_sym, start = None, 0
    sorted_sym = sym[order]
    for i in range(len(order)):
        if sorted_sym[i] != prev_sym:
            prev_sym, start = sorted_sym[i], i
        pos_in_group[order[i]] = i - start
        group_start[order[i]] = start
    for r in range(n):
        p, st = pos_in_group[r], group_start[r]
        for k in range(SEQ_L):
            j = p - (SEQ_L - 1 - k)
            seq[r, k] = order[st + max(j, 0)]
    return seq


def make_model(arch, nfeat, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    if arch == "mlp":
        return nn.Sequential(
            nn.Linear(nfeat, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 1))

    class GRUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(nfeat, 96, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(96, 1))

        def forward(self, x):
            h, _ = self.gru(x)
            return self.head(h[:, -1])
    return GRUNet()


def train_block(arch, X, y, seq, fit_idx, es_idx, te_idx, seed, threads):
    import torch
    torch.set_num_threads(threads)
    nfeat = X.shape[1]
    model = make_model(arch, nfeat, seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    seqt = torch.from_numpy(seq) if arch == "gru" else None

    def batch_x(rows):
        if arch == "gru":
            return Xt[seqt[rows].reshape(-1)].reshape(len(rows), SEQ_L, nfeat)
        return Xt[rows]

    def eval_loss(rows):
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(rows), 16384):
                b = rows[i:i + 16384]
                p = model(batch_x(b)).squeeze(-1)
                tot += float(((p - yt[b]) ** 2).sum())
                cnt += len(b)
        return tot / max(cnt, 1)

    rng = np.random.RandomState(seed)
    best, best_state, bad = np.inf, None, 0
    for ep in range(20):
        model.train()
        perm = rng.permutation(fit_idx)
        for i in range(0, len(perm), 8192):
            b = torch.from_numpy(perm[i:i + 8192])
            opt.zero_grad()
            p = model(batch_x(b)).squeeze(-1)
            loss = ((p - yt[b]) ** 2).mean()
            loss.backward()
            opt.step()
        vl = eval_loss(es_idx) if len(es_idx) else eval_loss(fit_idx[:50000])
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(te_idx), 16384):
            b = te_idx[i:i + 16384]
            outs.append(model(batch_x(b)).squeeze(-1).numpy())
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "gru"], required=True)
    ap.add_argument("--index", default="csi500", help="域")
    ap.add_argument("--target", default="y5d_rank")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--threads", type=int, default=40)
    args = ap.parse_args()

    ds, feats = load_ds(args.index)
    if args.arch == "gru":
        feats = [f for f in feats if f.endswith("|mean")]
    all_dates = sorted(ds["date"].unique())
    dev_dates = [d for d in all_dates if not C.is_test_date(d)]
    C.assert_dev_dates(dev_dates, "train_nn")
    ds = ds[ds["date"].isin(dev_dates)].reset_index(drop=True)

    date_to_idx = {d: i for i, d in enumerate(dev_dates)}
    didx = ds["date"].map(date_to_idx).values.astype(np.int32)
    X = np.nan_to_num(ds[feats].values.astype(np.float32), nan=0.0)
    y = ds[args.target].values.astype(np.float32)
    valid = ~np.isnan(y)
    seq = build_seq_index(ds) if args.arch == "gru" else None

    sched = schedule(dev_dates)
    print(f"nn-{args.arch} rows={len(ds)} feats={len(feats)} refits={len(sched)}", flush=True)

    out = []
    for bi, (tr_end, block) in enumerate(sched):
        t0 = time.time()
        tr_mask = (didx < tr_end) & valid
        te_mask = np.isin(didx, [date_to_idx[d] for d in block])
        if tr_mask.sum() < 1000 or te_mask.sum() == 0:
            continue
        uniq = np.unique(didx[tr_mask])
        es_days = set(uniq[-63:]) if len(uniq) > 200 else set()
        gap_days = set(uniq[-63 - C.EMBARGO_DAYS:-63]) if len(uniq) > 200 else set()
        rows = np.where(tr_mask)[0]
        drow = didx[rows]
        es_idx = rows[np.isin(drow, list(es_days))] if es_days else np.array([], np.int64)
        fit_idx = rows[~np.isin(drow, list(es_days | gap_days))]
        te_idx = np.where(te_mask)[0]
        preds = [train_block(args.arch, X, y, seq, fit_idx, es_idx, te_idx, s, args.threads)
                 for s in range(args.seeds)]
        s = np.mean(preds, 0)
        blk = ds.loc[te_mask, ["date", "symbol"]].copy()
        blk["score"] = s.astype(np.float32)
        out.append(blk)
        print(f"block {bi+1}/{len(sched)} {block[0]}..{block[-1]} "
              f"train={tr_mask.sum()} {time.time()-t0:.0f}s", flush=True)

    preds = pd.concat(out, ignore_index=True)
    fp = os.path.join(pred_dir(args.index), f"nn{args.arch}_{args.target}.parquet")
    preds.to_parquet(fp, index=False)
    print("saved", fp, len(preds), flush=True)


if __name__ == "__main__":
    main()
