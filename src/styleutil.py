# -*- coding: utf-8 -*-
"""风格因子构建与横截面中性化(项目A/B 共用).

从项目B 的 style_neutralize.py 抽出, 使纯多头选股(项目A)也能做风格中性口径。

风格因子全部只用**决策时点已知**的信息:
  size    log(流通市值), t−1 日口径(t日收盘市值含当日收盘价, 14:55 未知)
  vol20   过去20日已实现波动
  beta20  过去20日对域内等权收益的 beta
  mom20   过去20日累计收益
  rev5    过去5日累计收益(短期反转 —— 微观结构因子最可能的混淆项)
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

MCAP_ROOT = "${IDXML_MCAP_ROOT}"
PRICE_DAILY = os.environ.get("IDXML_PRICE_DAILY","")
STYLES = ["size", "vol20", "beta20", "mom20", "rev5"]


_LAST_SHARES = None      # 市值源末日的流通股本(用于缺失期重建), 惰性加载


def _last_known_shares():
    """${IDXML_MCAP_ROOT} 只到 2025-07-17; 之后用**最后已知流通股本 × 当日收盘价**重建市值。
    股本变动缓慢, 这是合理近似; 且 size 只用于风格中性化的横截面回归, 对量纲不敏感。"""
    global _LAST_SHARES
    if _LAST_SHARES is None:
        import glob
        fs = sorted(glob.glob(os.path.join(MCAP_ROOT, "*.parquet")))
        d = pd.read_parquet(fs[-1], columns=["code", "circulating_cap"])
        _LAST_SHARES = d.rename(columns={"code": "symbol", "circulating_cap": "shares"})
    return _LAST_SHARES


def load_mcap(date):
    fp = os.path.join(MCAP_ROOT, f"{date}.parquet")
    if not os.path.exists(fp):
        # 市值源缺失 → 用 最后已知流通股本 × 当日收盘价 重建
        pp = os.path.join(PRICE_DAILY, f"{date}.parquet")
        if not os.path.exists(pp):
            return None
        px = pd.read_parquet(pp, columns=["code", "close"])
        px["symbol"] = px["code"].str[:6]
        m = px.merge(_last_known_shares(), on="symbol", how="inner")
        m["mcap"] = m["close"] * m["shares"] / 1e4      # 与原口径同为"亿元"量级
        m["date"] = date
        return m[["symbol", "mcap", "date"]]
    d = pd.read_parquet(fp, columns=["code", "circulating_market_cap"])
    d = d.rename(columns={"code": "symbol", "circulating_market_cap": "mcap"})
    d["date"] = date
    return d


def build_styles(ds_dir, dates_needed):
    """从收益历史构建风格因子. 全部用 t−1 及更早的信息 → 无前视."""
    ds = pd.read_parquet(os.path.join(ds_dir, "ds.parquet"),
                         columns=["date", "symbol", "ret_fwd_1d"])
    ds["date"] = ds["date"].astype(str)
    ds = ds.sort_values(["symbol", "date"])
    g = ds.groupby("symbol")["ret_fwd_1d"]
    # ret_fwd_1d(t) 是 t→t+1 收益; 决策点 t 已知的最近一个完整日收益是 ret_fwd_1d(t−1)
    ds["r_lag"] = g.shift(1)
    gl = ds.groupby("symbol")["r_lag"]
    ds["vol20"] = gl.transform(lambda s: s.rolling(20, min_periods=10).std())
    ds["mom20"] = gl.transform(lambda s: s.rolling(20, min_periods=10).sum())
    ds["rev5"] = gl.transform(lambda s: s.rolling(5, min_periods=3).sum())
    # beta: 对域内等权收益
    mkt = ds.groupby("date")["r_lag"].transform("mean")
    ds["_x"] = mkt
    ds["_xy"] = ds["r_lag"] * mkt
    ds["_xx"] = mkt * mkt
    gg = ds.groupby("symbol")
    cov = gg["_xy"].transform(lambda s: s.rolling(20, min_periods=10).mean()) - \
        gg["r_lag"].transform(lambda s: s.rolling(20, min_periods=10).mean()) * \
        gg["_x"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    var = gg["_xx"].transform(lambda s: s.rolling(20, min_periods=10).mean()) - \
        gg["_x"].transform(lambda s: s.rolling(20, min_periods=10).mean()) ** 2
    ds["beta20"] = cov / var.replace(0, np.nan)

    # 市值: 用 t−1 日口径(t日收盘市值含当日收盘价, 决策点14:55严格说未知)
    alldates = sorted(set(ds["date"]) & set(dates_needed) | set(dates_needed))
    with ProcessPoolExecutor(30) as ex:
        mc = [r for r in ex.map(load_mcap, sorted(set(dates_needed))) if r is not None]
    mc = pd.concat(mc, ignore_index=True)
    mc = mc.sort_values(["symbol", "date"])
    mc["mcap_lag"] = mc.groupby("symbol")["mcap"].shift(1)
    mc["size"] = np.log(mc["mcap_lag"].clip(lower=1e-6))
    out = ds[["date", "symbol", "vol20", "beta20", "mom20", "rev5"]].merge(
        mc[["date", "symbol", "size"]], on=["date", "symbol"], how="left")
    return out


def neutralize(df, score_col="score"):
    """逐日横截面 OLS 取残差; 返回 (中性化分数, 平均R²)."""
    resid, r2s = [], []
    for d, g in df.groupby("date", sort=True):
        y = g[score_col].values.astype(float)
        X = g[STYLES].values.astype(float)
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        r = pd.Series(np.nan, index=g.index)
        if ok.sum() > len(STYLES) + 30:
            Xs = X[ok]
            # 风格先做横截面标准化, 避免量纲主导
            Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12)
            A = np.column_stack([np.ones(ok.sum()), Xs])
            yy = y[ok] - y[ok].mean()
            coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
            fit = A @ coef
            r.iloc[np.where(ok)[0]] = yy - fit
            ss = float(((yy - fit) ** 2).sum()), float((yy ** 2).sum())
            r2s.append(1 - ss[0] / (ss[1] + 1e-12))
        resid.append(r)
    return pd.concat(resid).sort_index(), float(np.mean(r2s)) if r2s else np.nan


