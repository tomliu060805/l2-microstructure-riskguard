# -*- coding: utf-8 -*-
"""补跑多种子分布所需的单种子模型(四层稳健性协议的第四层).

`train_baselines.py --seeds N` 固定用 range(N) 做平均, 选不了单个种子;
`seed_study.py` 又要求 `lgbm_y5d_rank_s{seed}.parquet` 这种单种子产物。本脚本补上这一环。

调度/embargo/超参与主线逐行一致(直接复用 fit_predict_lgbm), 唯一变量是 random_state。
用法: /usr/bin/python3 train_single_seeds.py --index csi1000 --seed-lo 0 --seed-hi 5 --threads 8
"""
import argparse, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE,"..","..","src"))
sys.path.insert(0, os.path.join(HERE,"..","..","longonly_falsified","code"))
import common as C
from dsutil import load_ds, schedule, pred_dir
from train_baselines import fit_predict_lgbm

ap=argparse.ArgumentParser()
ap.add_argument("--index",required=True)
ap.add_argument("--seed-lo",type=int,default=0)
ap.add_argument("--seed-hi",type=int,default=5)
ap.add_argument("--threads",type=int,default=8)
a=ap.parse_args()

ds,feats=load_ds(a.index)
dev=[d for d in sorted(ds["date"].unique()) if not C.is_test_date(d)]
C.assert_dev_dates(dev,"train_single_seeds")
ds=ds[ds["date"].isin(dev)].reset_index(drop=True)
d2i={d:i for i,d in enumerate(dev)}
didx=ds["date"].map(d2i).values.astype(np.int32)
X=ds[feats].values.astype(np.float32)
y=ds["y5d_rank"].values.astype(np.float32); valid=~np.isnan(y)
sched=schedule(dev)
print(f"{a.index} rows={len(ds)} feats={len(feats)} refits={len(sched)} seeds={a.seed_lo}..{a.seed_hi-1}",flush=True)

for s in range(a.seed_lo,a.seed_hi):
    fp=os.path.join(pred_dir(a.index),f"lgbm_y5d_rank_s{s}.parquet")
    if os.path.exists(fp):
        print(f"  seed {s} 已存在, 跳过",flush=True); continue
    t0=time.time(); out=[]
    for bi,(tr_end,block) in enumerate(sched):
        tr=(didx<tr_end)&valid; te=np.isin(didx,[d2i[d] for d in block])
        if tr.sum()<1000 or te.sum()==0: continue
        p=fit_predict_lgbm(X[tr],y[tr],X[te],didx[tr],[s],a.threads)
        blk=ds.loc[te,["date","symbol"]].copy(); blk["score"]=p.astype(np.float32)
        out.append(blk)
    pd.concat(out,ignore_index=True).to_parquet(fp,index=False)
    print(f"  seed {s} 完成 {time.time()-t0:.0f}s → {os.path.basename(fp)}",flush=True)
