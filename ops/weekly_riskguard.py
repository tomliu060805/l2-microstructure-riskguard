# -*- coding: utf-8 -*-
"""每周风控规避清单 —— 生产脚本.

产出「本周不建议持有」的股票排名。信号在交易日 14:55 可得, 标签期 5 个交易日,
因此每周固定跑一次、清单覆盖其后约一周持仓期, 与研究口径一致。

与研究代码的三点差异(都是生产该有的, 不是放松):
  1. **训练用全部可用历史**, 含原 test 段。test 段是"研究纪律"的封锁对象, 不是生产的;
     它已在 2026-08 一次性开封评估完毕, 继续封锁只会让生产模型少用两年数据。
  2. **重训按 63 交易日节奏**(与研究的 REFIT_EVERY 一致), 不是每周重训。
     每周重训会让模型在没有新增信息的情况下漂移, 且与回测口径不符。
  3. **只预测不评估** —— 没有前向标签可用, 脚本不产出任何收益数字。

用法:
    /usr/bin/python3 ops/weekly_riskguard.py                 # 自动取最新可用日
    /usr/bin/python3 ops/weekly_riskguard.py --as-of 2025-12-31 --top 200
    /usr/bin/python3 ops/weekly_riskguard.py --force-refit   # 强制重训

输出: ops/output/avoid_<as_of>.csv / .md, 并更新 ops/output/latest.csv
"""
import argparse
import datetime as dt
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "longonly_falsified", "code"))

import common as C                                   # noqa: E402
from dsutil import load_ds, pred_dir                 # noqa: E402
from train_baselines import fit_predict_lgbm         # noqa: E402

OUT_DIR = os.path.join(HERE, "output")
STATE_DIR = os.path.join(HERE, "state")
DELIVERED = ["csi1000", "gz2000"]        # 冻结配置选定的交付域; csi500/hs300 已判负, 不入生产
REFIT_EVERY = 63                          # 与 dsutil.REFIT_EVERY 一致


# ─────────────────────────── 数据新鲜度 ───────────────────────────
def factor_store_max_date():
    """因子库实际覆盖到哪天 —— 取所有因子目录的最小值(木桶效应: 任一因子缺就不能算)。"""
    if not C.FACTOR_ROOT or not os.path.isdir(C.FACTOR_ROOT):
        return None
    last = []
    for f in sorted(os.listdir(C.FACTOR_ROOT)):
        d = os.path.join(C.FACTOR_ROOT, f)
        if not os.path.isdir(d):
            continue
        files = [x[:-8] for x in os.listdir(d) if x.endswith(".parquet")]
        if files:
            last.append(max(files))
    return min(last) if last else None


def freshness_report(as_of):
    """把数据新鲜度显式打出来。

    这一段不是装饰 —— 一个定时任务最危险的失败模式, 是数据源停更而任务照常成功,
    每周输出一份看起来正常、实际早已过期的清单。所以陈旧必须是**响亮的**。
    """
    today = dt.date.today().isoformat()
    fmax = factor_store_max_date()
    lines = [f"运行日 {today} / 信号日 {as_of} / 因子库最新 {fmax}"]
    stale = False
    if fmax:
        gap = np.busday_count(np.datetime64(fmax), np.datetime64(today))
        lines.append(f"因子库滞后运行日 {gap} 个工作日")
        if gap > 10:
            stale = True
            lines.append(
                f"!! 数据陈旧: 因子库停在 {fmax}, 距今 {gap} 个工作日。"
                f"本次输出反映的是 {as_of} 的信号, **不是当周信号**。")
    return "\n".join(lines), stale


def factor_source_fingerprint():
    """因子源指纹 = 根路径 + 因子名集合的哈希。

    为什么需要它: 同一台机器上存在多个因子库版本(v1/v2), 因子**名字完全一样**
    但 248 个里有 129 个的**取值口径不同**, 个别相关性低到 0.002。
    换源不会报错、不会缺列, 模型照样出分, 清单照样生成 —— 只是全错。
    所以把源钉进模型状态, 对不上就拒绝预测。
    """
    import hashlib
    names = []
    if C.FACTOR_ROOT and os.path.isdir(C.FACTOR_ROOT):
        names = sorted(x for x in os.listdir(C.FACTOR_ROOT)
                       if os.path.isdir(os.path.join(C.FACTOR_ROOT, x)))
    h = hashlib.sha256(("|".join(names)).encode()).hexdigest()[:16]
    return {"root": C.FACTOR_ROOT, "n_factors": len(names), "names_sha": h}


# ─────────────────────────── 模型状态 ───────────────────────────
def state_path(index):
    return os.path.join(STATE_DIR, f"model_{index}.pkl")


def need_refit(index, dates, as_of, force):
    """自上次拟合起经过 >= REFIT_EVERY 个交易日才重训。"""
    p = state_path(index)
    if force or not os.path.exists(p):
        return True, None, "首次拟合" if not os.path.exists(p) else "强制重训"
    with open(p, "rb") as fh:
        st = pickle.load(fh)
    if st.get("feature_names") is None or st.get("factor_source") is None:
        return True, None, "旧状态缺特征名或因子源指纹"
    try:
        elapsed = dates.index(as_of) - dates.index(st["fit_date"])
    except ValueError:
        return True, st, "信号日或拟合日不在日历内"
    if elapsed >= REFIT_EVERY:
        return True, st, f"距上次拟合 {elapsed} 个交易日 >= {REFIT_EVERY}"
    return False, st, f"距上次拟合 {elapsed} 个交易日, 沿用"


def fit_model(index, ds, feats, dates, as_of, threads, seeds):
    """在 as_of 之前的全部有标签样本上拟合。

    embargo: 训练集只用到 as_of 前 EMBARGO_DAYS 天, 使最晚样本的 5 日标签窗
    结束于 as_of 前一日 —— 与研究口径逐行一致。
    """
    import lightgbm as lgb
    i_asof = dates.index(as_of)
    tr_end = max(0, i_asof - C.EMBARGO_DAYS)
    tr_dates = set(dates[:tr_end])

    m = ds["date"].astype(str).isin(tr_dates).values
    y = ds["y5d_rank"].values.astype(np.float32)
    m &= ~np.isnan(y)
    if m.sum() < 50_000:
        raise RuntimeError(f"{index}: 可用训练样本仅 {m.sum()}, 过少")

    X = ds.loc[m, feats].values.astype(np.float32)
    yy = y[m]
    didx = pd.factorize(ds.loc[m, "date"].astype(str))[0].astype(np.int32)

    # 早停切分与研究一致(训练窗末 63 天做 es, 中间留 embargo)
    uniq = np.unique(didx)
    es_days = set(uniq[-63:])
    gap_days = set(uniq[-63 - C.EMBARGO_DAYS:-63])
    es_mask = np.isin(didx, list(es_days))
    fit_mask = ~np.isin(didx, list(es_days | gap_days))

    models = []
    for s in seeds:
        mdl = lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, num_leaves=63,
            min_child_samples=200, feature_fraction=0.6, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=1.0, random_state=s, n_jobs=threads,
            verbose=-1)
        mdl.fit(X[fit_mask], yy[fit_mask],
                eval_set=[(X[es_mask], yy[es_mask])],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        models.append(mdl)

    return {"models": models, "fit_date": as_of, "feature_names": list(feats),
            "factor_source": factor_source_fingerprint(),
            "n_train": int(m.sum()), "train_end": dates[tr_end - 1],
            "seeds": list(seeds), "fitted_at": dt.datetime.now().isoformat(timespec="seconds")}


def predict(st, ds, feats, as_of):
    day = ds[ds["date"].astype(str) == as_of]
    if day.empty:
        return None
    if list(feats) != list(st["feature_names"]):
        raise RuntimeError("特征名与模型状态不一致, 需重训(--force-refit)")
    cur, was = factor_source_fingerprint(), st.get("factor_source")
    if was and (cur["root"] != was["root"] or cur["names_sha"] != was["names_sha"]):
        raise RuntimeError(
            f"因子源与训练时不一致 —— 拒绝预测。\n"
            f"  训练时: {was['root']} ({was['n_factors']} 因子, sha {was['names_sha']})\n"
            f"  当前:   {cur['root']} ({cur['n_factors']} 因子, sha {cur['names_sha']})\n"
            f"  因子名相同不代表口径相同(v1/v2 有 129/248 取值不同, 最低相关 0.002)。\n"
            f"  换源必须重建面板+数据集并全量重训, 不能只 --force-refit。")
    X = day[feats].values.astype(np.float32)
    p = np.mean([m.predict(X) for m in st["models"]], axis=0)
    out = day[["symbol", "weight", "tradable"]].copy()
    out["score"] = p.astype(np.float32)
    # 域内百分位: 0 = 最差。跨域可比的唯一口径 —— 原始 score 是域内秩目标, 不可跨域比大小。
    out["pct"] = out["score"].rank(pct=True)
    return out.sort_values("score").reset_index(drop=True)


# ─────────────────────────── 主流程 ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None, help="信号日; 默认取最新可用交易日")
    ap.add_argument("--top", type=int, default=200, help="输出排名最差的前 N 只")
    ap.add_argument("--indexes", default=",".join(DELIVERED))
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3, help="种子平均数(研究口径为3)")
    ap.add_argument("--force-refit", action="store_true")
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    indexes = [x.strip() for x in a.indexes.split(",") if x.strip()]

    t0 = time.time()
    per_index, meta = {}, {}
    as_of = a.as_of

    for ix in indexes:
        ds, feats = load_ds(ix)
        dates = sorted(ds["date"].astype(str).unique().tolist())
        cur = as_of or dates[-1]
        if cur not in dates:
            raise SystemExit(f"{ix}: 信号日 {cur} 不在数据集内(最新 {dates[-1]})")
        as_of = cur

        refit, st, why = need_refit(ix, dates, as_of, a.force_refit)
        print(f"[{ix}] {why}", flush=True)
        if refit:
            t1 = time.time()
            st = fit_model(ix, ds, feats, dates, as_of, a.threads, range(a.seeds))
            with open(state_path(ix), "wb") as fh:
                pickle.dump(st, fh)
            print(f"[{ix}] 拟合完成 {time.time()-t1:.0f}s  "
                  f"样本={st['n_train']:,}  训练截至={st['train_end']}", flush=True)

        pr = predict(st, ds, feats, as_of)
        if pr is None or pr.empty:
            print(f"[{ix}] !! {as_of} 无截面数据, 跳过", flush=True)
            continue
        pr["index"] = ix
        per_index[ix] = pr
        meta[ix] = {"n_universe": len(pr), "fit_date": st["fit_date"],
                    "train_end": st["train_end"], "n_train": st["n_train"]}
        print(f"[{ix}] 截面 {len(pr)} 只, 已打分", flush=True)

        del ds

    if not per_index:
        raise SystemExit("没有任何域产出预测")

    merged = pd.concat(per_index.values(), ignore_index=True)

    # ── 跨域去重 ──
    # csi1000 与 gz2000 **有重叠**(gz2000 是市值秩 1001-3000 的重构域, 2025-12-31 实测
    # 与 csi1000 交集 612 只 = 后者的 61%)。不去重的话同一只股票占两个名额,
    # "前200"实际只有 170 余只。common.py 里"四指数互斥"说的是四个 CSI 官方指数,
    # 不含重构的 gz2000 —— 这是个容易踩的坑。
    #
    # 去重规则: 取该股在各域中**最差**的 pct(更保守), 并记录是否两域都判它差 ——
    # 两个独立截面同时把一只股票排到尾部, 是比单域更强的证据。
    merged["n_dom"] = merged.groupby("symbol")["index"].transform("size")
    merged["doms"] = merged.groupby("symbol")["index"].transform(lambda g: "+".join(sorted(g)))
    merged = (merged.sort_values("pct")
                    .drop_duplicates(subset="symbol", keep="first")
                    .reset_index(drop=True))
    merged.insert(0, "rank", np.arange(1, len(merged) + 1))
    top = merged.head(a.top).copy()

    fresh_txt, stale = freshness_report(as_of)

    csv_p = os.path.join(OUT_DIR, f"avoid_{as_of}.csv")
    top.to_csv(csv_p, index=False, encoding="utf-8-sig")
    top.to_csv(os.path.join(OUT_DIR, "latest.csv"), index=False, encoding="utf-8-sig")

    # ── 报告 ──
    L = [f"# 风控规避清单 — 信号日 {as_of}", ""]
    if stale:
        L += ["> ## !! 数据陈旧警告", ">",
              "> " + fresh_txt.replace("\n", "\n> "), ""]
    else:
        L += ["```", fresh_txt, "```", ""]
    L += [f"**建议在 {as_of} 收盘后调仓, 清单覆盖其后约 5 个交易日。**", "",
          "口径: 持有指数全部成分、按基准权重, 将下列个股权重置零。",
          "分数为域内横截面秩预测, `pct` 是**域内**百分位(越小越差); 跨域只能比 pct, 不能比 score。", ""]
    L += ["| 域 | 截面 | 拟合日 | 训练截至 | 训练样本 |", "|---|---|---|---|---|"]
    for ix, m in meta.items():
        L.append(f"| {ix} | {m['n_universe']} | {m['fit_date']} | {m['train_end']} | {m['n_train']:,} |")
    L += ["", f"## 排名最差 {len(top)} 只", "",
          "| # | 代码 | 判它差的域 | 域内百分位 | 基准权重% | 可交易 |", "|---|---|---|---|---|---|"]
    for r in top.itertuples():
        dom = f"**{r.doms}**" if r.n_dom > 1 else r.doms
        L.append(f"| {r.rank} | {r.symbol} | {dom} | {r.pct*100:.2f}% | "
                 f"{r.weight*100:.3f} | {'是' if r.tradable else '**否**'} |")
    n_untradable = int((~top["tradable"].astype(bool)).sum())
    n_both = int((top["n_dom"] > 1).sum())
    L += ["", f"其中 {n_untradable} 只在信号日不可交易(停牌/跌停封板), 无法当日卖出, 需顺延。",
          f"{n_both} 只被**两个域同时**判为尾部(表中域名加粗) —— 两个独立截面一致, 证据更强。",
          f"两域成分有重叠, 已按代码去重(取最差 pct); 去重前候选 {len(merged)} 只。", ""]
    md_p = os.path.join(OUT_DIR, f"avoid_{as_of}.md")
    with open(md_p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    print(f"\n{fresh_txt}")
    print(f"\n输出: {csv_p}\n      {md_p}\n耗时 {time.time()-t0:.0f}s")
    if stale:
        print("\n!! 因子库陈旧 —— 本清单不代表当周信号, 见报告顶部警告。")
        sys.exit(3)          # 非零退出码, 让 cron 的失败告警能抓到


if __name__ == "__main__":
    main()
