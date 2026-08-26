# -*- coding: utf-8 -*-
"""项目B 复查: 用姊妹项目新造的两把尺子重量已交付结论.

## 为什么复查
姊妹项目 `风格解耦选股` 用两把新尺子把自己整条主线判负了:
  ① **Barra+行业标尺**(原来只有 size/vol20/beta20/mom20/rev5 五个自制风格, 缺行业与流动性)
  ② **随机组合零基准**(正确的零假设不是指数, 是同等构建方式的随机组合)
项目B 是**已开封、已写进简历**的成果, 但这两把尺子从未量过它 —— 必须复查。

## 一个结构性差异(先说清楚, 决定了预期)
姊妹项目死于**等权 vs 市值加权的规模溢价**: 它的组合是**等权 top decile**, 对比市值加权指数,
中间嵌着未控制的规模押注(随机挑50只等权就能跑赢指数 5~7%/年)。
**项目B 不同**: 它是**按基准权重持有全部成分、剔除最差 10%**(`wb = weight/Σweight`),
**没有等权押注** ⇒ 那个杀手在这里不适用。本脚本要验证的是另外两件事:
  · 换 Barra+行业标尺后, 超额还剩多少
  · 对"**随机剔除 10%**"的零分布, 真实剔除是否显著更好

## 口径
只在**验证段**做(测试段已开封一次, 此处不再动它)。csi1000 与 gz2000 两域。
"""
import os, sys, warnings, glob
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,"..","..","src")); sys.path.insert(0,HERE)
# barra_styles 已收进本仓 src/(原出自姊妹项目 风格解耦选股), 保持自包含
import common as C
from dsutil import ds_dir, pred_dir
from short_side_exclude import build, simulate
from styleutil import build_styles as build_styles_old, STYLES as S5
from barra_styles import build as build_barra, NUM as SB   # ← src/barra_styles.py
OUT=os.path.join(HERE,"..","results","复查_2026")
NDRAW=200; FRAC=0.10; BP=20


def neu(m,cols,ind,col="score"):
    res,r2s=[],[]
    for d,g in m.groupby("date",sort=True):
        y=g[col].values.astype(float); X=g[cols].values.astype(float)
        ok=np.isfinite(y)&np.isfinite(X).all(1)
        if ind: ok&=g[ind].notna().values
        r=pd.Series(np.nan,index=g.index)
        if ok.sum()>len(cols)+40:
            Z=X[ok]; Z=(Z-Z.mean(0))/(Z.std(0)+1e-12)
            A=[np.ones(ok.sum()),Z]
            if ind:
                dm=pd.get_dummies(g.loc[ok,ind].astype(str),drop_first=True).values.astype(float)
                if dm.shape[1]: A.append(dm)
            A=np.column_stack(A); yy=y[ok]-y[ok].mean()
            c,*_=np.linalg.lstsq(A,yy,rcond=None); fit=A@c
            r.iloc[np.where(ok)[0]]=yy-fit
            r2s.append(1-((yy-fit)**2).sum()/((yy**2).sum()+1e-12))
        res.append(r)
    return pd.concat(res).sort_index(), float(np.mean(r2s)) if r2s else np.nan


def run(df,col,frac=FRAC,bp=BP):
    d=df.copy(); d["score"]=d[col]
    d=d.sort_values(["symbol","date"])
    d["score_sm"]=d.groupby("symbol")["score"].transform(lambda s:s.rolling(5,min_periods=1).mean())
    d=d.sort_values(["date","symbol"]).reset_index(drop=True)
    res,te,tw,n=simulate(d,frac,"smooth",cost_bps=(bp,))
    r=res[bp]
    return dict(ann=r["ann"]*100,ir=r["ir"],wr=r["wr"],payoff=r["payoff"],pf=r["pf"],te=te*100,turnover=tw,n=n)


def main():
    rows=[]
    for idx in ["csi1000","gz2000"]:
        print(f"══ {idx}",flush=True)
        df,dates=build(idx,pred_name="lgbm_y5d_rank")
        st5=build_styles_old(ds_dir(idx),dates)
        B=build_barra(ds_dir(idx),dates)
        m=df.merge(st5,on=["date","symbol"],how="left").merge(B,on=["date","symbol"],how="left",suffixes=("","_b"))
        res={}
        res["原始"]=(run(m,"score"),np.nan)
        m["n5"],r5=neu(m,S5,None); res["S0 5风格(原口径)"]=(run(m,"n5"),r5)
        m["nb"],rb=neu(m,SB,"ind"); res["BARRA+行业"]=(run(m,"nb"),rb)
        for k,(v,r2) in res.items():
            print(f"   {k:18s} 净{BP}bp {v['ann']:+.3f}% IR {v['ir']:5.2f} TE {v['te']:.2f}% "
                  f"| 触发{v['n']}日 胜率{v['wr']:.1%} 单笔均{v['ann']/242*100:+.2f}bp "
                  f"盈亏比{v['payoff']:.3f} 盈利因子{v['pf']:.3f} | R²{r2:.3f}",flush=True)
            rows.append(dict(域=idx,口径=k,净超额=v["ann"],IR=v["ir"],TE=v["te"],胜率=v["wr"],
                             盈亏比=v["payoff"],盈利因子=v["pf"],换手=v["turnover"],R2=r2))
        # 随机剔除零分布(按 BARRA 口径的构建方式, 只把分数换成随机)
        print(f"   造随机剔除零分布 ×{NDRAW} ...",flush=True)
        z=[]
        for i in range(NDRAW):
            rng=np.random.RandomState(3000+i); m["_r"]=rng.rand(len(m))
            z.append(run(m,"_r")["ann"])
        z=np.array([x for x in z if np.isfinite(x)])
        real=res["BARRA+行业"][0]["ann"]; pc=(z<real).mean()
        print(f"   随机剔除零分布: 均值 {z.mean():+.3f}%  sd {z.std():.3f}  p95 {np.percentile(z,95):+.3f}%")
        print(f"   ★真实(BARRA口径) {real:+.3f}%  超出零基准 {real-z.mean():+.3f}%  分位 {pc:.1%} "
              f"{'✓过p95' if pc>0.95 else '✗'}\n",flush=True)
        rows.append(dict(域=idx,口径="随机剔除零基准",净超额=z.mean(),R2=np.nan))
        rows.append(dict(域=idx,口径="★超出零基准(R2列=分位)",净超额=real-z.mean(),R2=pc))
    os.makedirs(OUT,exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT,"复查_Barra与随机零基准.csv"),index=False)


if __name__=="__main__": main()
