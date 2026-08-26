# -*- coding: utf-8 -*-
"""预测评估与回测: RankIC / 分组单调性 / 多空组合 / 换手与成本敏感性.

- 只评估 dev 段(common.assert_dev_dates 把关; 最终评估 IDXML_TEST_UNLOCK=1).
- 分组用确定性随机破并列(种子=日期), 保证各组样本数恒等(历史教训: 并列压组会造假).
- 组合: 明日持仓 = 今日收盘按 score 选 top/bottom decile 等权(仅 tradable=1 可开新仓),
  用 ret_fwd_1d 逐日复利; 成本 = Σ|Δw| × 单边费率, 报 0/10/20/30bp.
- 报表带: 触发数(交易日数)/胜率/单笔均(日均)/盈亏比/盈利因子.

用法: /usr/bin/python3 backtest.py --pred lgbm_y5d_rank [--eval-window val|dev]
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
RESULTS = os.path.join(HERE, "..", "results")
import common as C
from dsutil import ds_dir, pred_dir
from btutil import NQ, tie_broken_decile, spearman_ic

warnings.filterwarnings("ignore")








def build_weights(g, q, prev_w, style, ti, state):
    """按组合风格生成当日权重. g: 当日df(含symbol/tradable), q: 十分组(0低9高)."""
    tr = g["tradable"].values == 1
    syms = g["symbol"].values
    if style == "plain":
        top = set(syms[(q == NQ - 1) & tr])
        bot = set(syms[(q == 0) & tr])
        held = set(prev_w) & set(syms[~tr])
        top |= {s for s in held if prev_w.get(s, 0) > 0}
        bot |= {s for s in held if prev_w.get(s, 0) < 0}
        w = {s: 1.0 / max(len(top), 1) for s in top}
        for s in bot:
            w[s] = w.get(s, 0) - 1.0 / max(len(bot), 1)
        return w
    if style == "tranche5":
        # 5个子组合, 每天只重建1个(轮转), 每个子组合持有5天 → 换手≈1/5
        k = ti % 5
        subs = state.setdefault("subs", [dict() for _ in range(5)])
        top = set(syms[(q == NQ - 1) & tr])
        bot = set(syms[(q == 0) & tr])
        sub = {s: 1.0 / max(len(top), 1) for s in top}
        for s in bot:
            sub[s] = sub.get(s, 0) - 1.0 / max(len(bot), 1)
        subs[k] = sub
        w = {}
        for sb in subs:
            for s, wi in sb.items():
                w[s] = w.get(s, 0) + wi / 5.0
        return w
    if style == "buffer":
        # 进入=最优/最差10%, 退出=跌出20%缓冲带; 不可交易则维持原仓
        n = len(syms)
        rk = np.empty(n)
        order = np.argsort(q)  # q已是破并列分组; 用组序即可, 精细用score
        pct = (q + 0.5) / NQ
        longs = {s for s, p, t in zip(syms, pct, tr) if p > 0.9 and t}
        long_keep = {s for s, p in zip(syms, pct) if p > 0.8}
        shorts = {s for s, p, t in zip(syms, pct, tr) if p < 0.1 and t}
        short_keep = {s for s, p in zip(syms, pct) if p < 0.2}
        untr = set(syms[~tr])
        for s, wi in prev_w.items():
            if wi > 0 and (s in long_keep or s in untr):
                longs.add(s)
            if wi < 0 and (s in short_keep or s in untr):
                shorts.add(s)
        w = {s: 1.0 / max(len(longs), 1) for s in longs}
        for s in shorts:
            w[s] = w.get(s, 0) - 1.0 / max(len(shorts), 1)
        return w
    raise ValueError(style)


def build_long_weights(g, q, prev_w, style, ti, state):
    """纯多头: top decile 等权. tranche5 = 每天轮换1/5."""
    tr = g["tradable"].values == 1
    syms = g["symbol"].values
    top = set(syms[(q == NQ - 1) & tr])
    if style == "tranche5":
        k = ti % 5
        subs = state.setdefault("subs", [dict() for _ in range(5)])
        subs[k] = {s: 1.0 / max(len(top), 1) for s in top}
        w = {}
        for sb in subs:
            for s, wi in sb.items():
                w[s] = w.get(s, 0) + wi / 5.0
        return w
    held = set(prev_w) & set(syms[~tr])
    top |= held
    return {s: 1.0 / max(len(top), 1) for s in top}


def run(index, pred_name, eval_window, unlock, style="plain", side="ls"):
    preds = pd.read_parquet(os.path.join(pred_dir(index), f"{pred_name}.parquet"))
    lab = pd.read_parquet(os.path.join(ds_dir(index), "ds.parquet"),
                          columns=["date", "symbol", "ret_fwd_1d", "ret_fwd_5d", "tradable"])
    lab["date"] = lab["date"].astype(str)
    df = preds.merge(lab, on=["date", "symbol"], how="inner")

    dates = sorted(df["date"].unique())
    if eval_window == "val":
        dates = [d for d in dates if C.VAL_START <= d < C.TEST_START]
    elif eval_window == "test":
        dates = [d for d in dates if C.is_test_date(d)]
    if not unlock:
        C.assert_dev_dates(dates, f"backtest:{pred_name}")
    df = df[df["date"].isin(dates)]

    ics_1d, ics_5d = [], []
    dec_ret = np.zeros((len(dates), NQ))
    ls_gross, turnover, bench = [], [], []
    prev_w, state = {}, {}
    for ti, (d, g) in enumerate(df.groupby("date", sort=True)):
        g = g.dropna(subset=["score"])
        ics_1d.append(spearman_ic(g["score"], g["ret_fwd_1d"]))
        ics_5d.append(spearman_ic(g["score"], g["ret_fwd_5d"]))
        q = tie_broken_decile(g["score"].values, d)
        r1 = g["ret_fwd_1d"].values
        for k in range(NQ):
            dec_ret[ti, k] = np.nanmean(r1[q == k])
        if side == "long":
            w = build_long_weights(g, q, prev_w, style, ti, state)
        else:
            w = build_weights(g, q, prev_w, style, ti, state)
        tw = sum(abs(w.get(s, 0) - prev_w.get(s, 0)) for s in set(w) | set(prev_w))
        turnover.append(tw)
        ret_map = dict(zip(g["symbol"].values, r1))
        ls_gross.append(sum(wi * ret_map.get(s, 0.0) for s, wi in w.items() if not np.isnan(ret_map.get(s, np.nan))))
        bench.append(np.nanmean(r1))
        prev_w = w

    ls = np.array(ls_gross)
    tw = np.array(turnover)
    ic1, ic5 = np.array(ics_1d), np.array(ics_5d)
    ann = 244

    def perf(r):
        mu, sd = np.nanmean(r), np.nanstd(r)
        eq = np.cumprod(1 + np.nan_to_num(r))
        dd = 1 - eq / np.maximum.accumulate(eq)
        pos, neg = r[r > 0], r[r < 0]
        return {
            "ann_ret": float(mu * ann), "ann_sharpe": float(mu / (sd + 1e-12) * np.sqrt(ann)),
            "maxDD": float(dd.max()),
            "n_days": int(len(r)), "win_rate": float((r > 0).mean()),
            "avg_daily_bp": float(mu * 1e4),
            "payoff": float(pos.mean() / (abs(neg.mean()) + 1e-12)) if len(neg) else np.inf,
            "profit_factor": float(pos.sum() / (abs(neg.sum()) + 1e-12)) if len(neg) else np.inf,
        }

    rep = {
        "pred": pred_name, "window": eval_window, "style": style,
        "dates": [dates[0], dates[-1]], "n_days": len(dates),
        "rank_ic_1d": {"mean": float(np.nanmean(ic1)), "std": float(np.nanstd(ic1)),
                       "icir_daily": float(np.nanmean(ic1) / (np.nanstd(ic1) + 1e-12)),
                       "pct_pos": float((ic1 > 0).mean())},
        "rank_ic_5d": {"mean": float(np.nanmean(ic5)), "std": float(np.nanstd(ic5)),
                       "icir_daily": float(np.nanmean(ic5) / (np.nanstd(ic5) + 1e-12)),
                       "pct_pos": float((ic5 > 0).mean()),
                       "note": "5d重叠期限, 日IC自相关, ICIR偏乐观"},
        "decile_ann_ret": {f"D{k+1}": float(np.nanmean(dec_ret[:, k]) * ann) for k in range(NQ)},
        "turnover_oneway_daily": float(tw.mean() / 2),
        "LS_gross": perf(ls),
    }
    for bp in (10, 20, 30):
        rep[f"LS_net_{bp}bp"] = perf(ls - tw * bp / 1e4)
    if side == "long":
        b = np.array(bench)
        ex = ls - b
        te = float(np.nanstd(ex) * np.sqrt(ann))
        rep["side"] = "long_only_top_decile"
        rep["benchmark"] = "universe_equal_weight(每日成分等权, 复权c2c)"
        rep["bench_ann_ret"] = float(np.nanmean(b) * ann)
        rep["port_ann_ret"] = float(np.nanmean(ls) * ann)
        rep["excess_gross"] = dict(perf(ex), tracking_error=te,
                                   info_ratio=float(np.nanmean(ex) / (np.nanstd(ex) + 1e-12) * np.sqrt(ann)))
        for bp in (10, 20, 30):
            exn = ex - tw * bp / 1e4
            rep[f"excess_net_{bp}bp"] = dict(perf(exn),
                                             info_ratio=float(np.nanmean(exn) / (np.nanstd(exn) + 1e-12) * np.sqrt(ann)))
    return rep


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--index", required=True,

                    help="hs300/csi500/csi1000/csi2000 — 每个指数域独立建模")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--eval-window", default="val", choices=["val", "dev", "test"])
    ap.add_argument("--style", default="plain", choices=["plain", "tranche5", "buffer"])
    ap.add_argument("--side", default="ls", choices=["ls", "long"])
    args = ap.parse_args()
    unlock = os.environ.get("IDXML_TEST_UNLOCK") == "1"
    rep = run(args.index, args.pred, args.eval_window, unlock, style=args.style, side=args.side)
    os.makedirs(os.path.join(RESULTS, args.index), exist_ok=True)
    out = os.path.join(RESULTS, args.index,
                       f"bt_{args.pred}_{args.eval_window}_{args.style}_{args.side}.json")
    with open(out, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
