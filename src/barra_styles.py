# -*- coding: utf-8 -*-
"""Barra CNE5 口径风格集 + 行业 —— 替代原来的 5 风格.

## 为什么要换
原 5 风格(size/vol20/beta20/mom20/rev5)**不含行业、不含流动性**。实测(实验3诊断):
加上换手与行业后, baseline 的中性化 IR 0.61→0.29, featneu_clean 1.31→**0.82**
—— 原口径系统性高估了"干净程度", 所有历史绝对数字都要按本口径重报。

## 因子清单(11 数值 + 行业哑变量)
| 名称 | 口径 | Barra 对应 |
|---|---|---|
| size      | log 流通市值                        | LNCAP |
| nlsize    | 标准化 size 的三次方, 对 size 正交   | NLSIZE |
| beta20/60 | 对域内等权收益的 20/60 日 beta       | BETA |
| mom20/120 | 20/120 日累计收益                    | MOMENTUM |
| rev5      | 5 日累计收益(A股短反转, 保留)       | —— |
| vol20     | 20 日已实现波动                      | RESVOL |
| liq       | log(20日均换手率)                    | LIQUIDITY |
| bp        | 1/PB                                 | BOOK-TO-PRICE |
| ep        | 1/PE                                 | EARNINGS YIELD |
| lev       | 总负债/总资产                        | LEVERAGE |
| growth    | 归母净利同比                          | GROWTH |
| ind       | 申万一级(sw_l1) 哑变量               | INDUSTRY |

## ★PIT 纪律(与无前视审计一致)
决策点是 **d 日 14:55**, 故**全部风格只用到 d−1 收盘及更早**:
  · 行情派生(市值/换手/BP/EP): 取 **d−1** 当日的 valuation 快照
  · 收益派生(beta/mom/rev/vol): r_lag = ret_fwd_1d.shift(2) = close(t−1)/close(t−2)−1
  · 报表派生(lev/growth): d−1 快照且强制 pub_date <= d−1(共享源非PIT, 见审计)
  · 行业: d−1 快照

## 风险防护(用户明确要求注意)
  1. 每个风格逐日截面 **1%/99% 缩尾**后再标准化 —— 防极值主导回归
  2. 行业成员数 < MIN_IND 的并入 "OTHER" —— 防小格过拟合
  3. 落盘同时输出覆盖率与相关矩阵诊断 —— 共线性可查
"""
from __future__ import annotations
import os, glob, bisect
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

PD  = os.environ.get("IDXML_PRICE_DAILY", os.environ.get("IDXML_PRICE_DAILY",""))
VAL = os.environ.get("IDXML_VALUATION", os.environ.get("IDXML_VALUATION",""))
FIN = os.environ.get("IDXML_FININD", os.environ.get("IDXML_FININD",""))
BAL = os.environ.get("IDXML_BALANCE", os.environ.get("IDXML_BALANCE",""))
IND = os.environ.get("IDXML_INDUSTRY", os.environ.get("IDXML_INDUSTRY",""))

NUM = ["size","nlsize","beta20","beta60","mom20","mom120","rev5","vol20","liq","bp","ep","lev","growth"]
MIN_IND = 5          # 行业最小成员数, 不足并入 OTHER
WINS = 0.01          # 缩尾比例


def _val_one(d):
    fp=f"{VAL}/{d}.parquet"
    if not os.path.exists(fp): return None
    v=pd.read_parquet(fp,columns=["code","turnover_ratio","pe_ratio","pb_ratio","circulating_market_cap"])
    v["symbol"]=v["code"].astype(str).str[:6]
    v=v.drop_duplicates("symbol").set_index("symbol")
    with np.errstate(divide="ignore",invalid="ignore"):
        out=pd.DataFrame({
            "mcap":v["circulating_market_cap"].values,
            "turn":v["turnover_ratio"].values,
            "bp":np.where(np.abs(v["pb_ratio"])>1e-6,1/v["pb_ratio"],np.nan),
            "ep":np.where(np.abs(v["pe_ratio"])>1e-6,1/v["pe_ratio"],np.nan)},index=v.index)
    return d,out


def _fund_one(d):
    """报表派生: 强制 pub_date <= d(共享源快照本身非PIT, 见 docs/无前视审计.md)"""
    o={}
    fp=f"{FIN}/{d}.parquet"
    if os.path.exists(fp):
        f=pd.read_parquet(fp,columns=["code","pub_date","inc_net_profit_to_shareholders_year_on_year"])
        f=f[f["pub_date"].astype(str)<=d]
        f["symbol"]=f["code"].astype(str).str[:6]
        o["growth"]=f.drop_duplicates("symbol").set_index("symbol").iloc[:,-1]
    fp=f"{BAL}/{d}.parquet"
    if os.path.exists(fp):
        b=pd.read_parquet(fp,columns=["code","pub_date","total_assets","total_liability"])
        b=b[b["pub_date"].astype(str)<=d]
        b["symbol"]=b["code"].astype(str).str[:6]
        b=b.drop_duplicates("symbol").set_index("symbol")
        ta=b["total_assets"].values.astype(float)
        o["lev"]=pd.Series(np.where(ta>0,b["total_liability"].values/ta,np.nan),index=b.index)
    if not o: return None
    return d,pd.DataFrame(o)


def _ind_one(d):
    fp=f"{IND}/{d}.parquet"
    if not os.path.exists(fp): return None
    x=pd.read_parquet(fp,columns=["code","category","industry_code"])
    x=x[x["category"]=="sw_l1"]
    x["symbol"]=x["code"].astype(str).str[:6]
    return d,x.drop_duplicates("symbol").set_index("symbol")["industry_code"]


def winsor_z(a):
    """逐日截面 1%/99% 缩尾后标准化(风险防护1)"""
    a=np.asarray(a,float); ok=np.isfinite(a)
    if ok.sum()<20: return np.full_like(a,np.nan)
    lo,hi=np.nanquantile(a[ok],[WINS,1-WINS])
    b=np.clip(a,lo,hi)
    return (b-np.nanmean(b))/(np.nanstd(b)+1e-12)


def build(ds_dir_, dates_needed, workers=20, verbose=True):
    """返回 [date, symbol] + NUM + ind 的面板。全部只用 d−1 及更早。"""
    ds=pd.read_parquet(os.path.join(ds_dir_,"ds.parquet"),columns=["date","symbol","ret_fwd_1d"])
    ds["date"]=ds["date"].astype(str); ds["symbol"]=ds["symbol"].astype(str)
    ds=ds.sort_values(["symbol","date"])
    # 收益派生: shift(2) ⇒ 最新一条是 close(t−1)/close(t−2)−1, 不含 14:55 未知的 close(t)
    ds["r"]=ds.groupby("symbol")["ret_fwd_1d"].shift(2)
    g=ds.groupby("symbol")["r"]
    ds["vol20"]=g.transform(lambda s:s.rolling(20,min_periods=10).std())
    ds["mom20"]=g.transform(lambda s:s.rolling(20,min_periods=10).sum())
    ds["mom120"]=g.transform(lambda s:s.rolling(120,min_periods=60).sum())
    ds["rev5"]=g.transform(lambda s:s.rolling(5,min_periods=3).sum())
    mkt=ds.groupby("date")["r"].transform("mean")
    ds["_x"],ds["_xy"],ds["_xx"]=mkt,ds["r"]*mkt,mkt*mkt
    gg=ds.groupby("symbol")
    for w in (20,60):
        rm=lambda c: gg[c].transform(lambda s:s.rolling(w,min_periods=max(10,w//2)).mean())
        ds[f"beta{w}"]=(rm("_xy")-rm("r")*rm("_x"))/(rm("_xx")-rm("_x")**2).replace(0,np.nan)

    DD=sorted(os.path.basename(x)[:10] for x in glob.glob(f"{PD}/*.parquet"))
    need=sorted(set(dates_needed))
    prev={d:DD[bisect.bisect_left(DD,d)-1] for d in need if bisect.bisect_left(DD,d)>0}
    src=sorted(set(prev.values()))
    if verbose: print(f"  取 d−1 快照 {len(src)} 天 ...",flush=True)
    with ProcessPoolExecutor(workers) as ex:
        V={d:x for d,x in (r for r in ex.map(_val_one,src) if r)}
        F={d:x for d,x in (r for r in ex.map(_fund_one,src) if r)}
        I={d:x for d,x in (r for r in ex.map(_ind_one,src) if r)}

    recs=[]
    for d in need:
        p=prev.get(d)
        if p is None or p not in V: continue
        v=V[p].copy()
        f=F.get(p)
        if f is not None: v=v.join(f,how="left")
        for c in ("growth","lev"):
            if c not in v.columns: v[c]=np.nan
        v["size"]=np.log(v["mcap"].clip(lower=1e-6))
        v["liq"]=np.log(v["turn"].clip(lower=1e-6))
        t=v[["size","liq","bp","ep","lev","growth"]].copy()
        t["date"]=d; t["symbol"]=t.index
        ii=I.get(p)
        t["ind"]=ii.reindex(t.index).values if ii is not None else np.nan
        recs.append(t)
    X=pd.concat(recs,ignore_index=True)
    out=ds[["date","symbol","vol20","mom20","mom120","rev5","beta20","beta60"]].merge(
        X,on=["date","symbol"],how="inner")

    # 逐日: 缩尾标准化 + nlsize(对 size 正交) + 小行业并档(风险防护1&2)
    parts=[]
    for d,gd in out.groupby("date",sort=True):
        gd=gd.copy()
        for c in ["size","liq","bp","ep","lev","growth","vol20","mom20","mom120","rev5","beta20","beta60"]:
            gd[c]=winsor_z(gd[c].values)
        s=gd["size"].values
        nl=np.where(np.isfinite(s),s**3,np.nan)
        ok=np.isfinite(nl)&np.isfinite(s)
        if ok.sum()>20:                                   # 对 size 正交化
            b=np.polyfit(s[ok],nl[ok],1)
            nl=np.where(ok,nl-(b[0]*s+b[1]),np.nan)
        gd["nlsize"]=winsor_z(nl)
        cnt=gd["ind"].value_counts()
        small=set(cnt[cnt<MIN_IND].index)
        gd["ind"]=gd["ind"].where(~gd["ind"].isin(small),"OTHER").fillna("OTHER")
        parts.append(gd)
    R=pd.concat(parts,ignore_index=True)
    return R[["date","symbol"]+NUM+["ind"]]
