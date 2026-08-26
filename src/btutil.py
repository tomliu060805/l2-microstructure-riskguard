# -*- coding: utf-8 -*-
"""回测公共原语(两个子项目共用): 分组/IC/绩效指标.

分组用确定性随机破并列(种子=日期数字), 保证各组样本数恒等 ——
历史教训: L2因子大量取值并列, 直接按秩分组会把整块并列压进同一组, 造出假收益。
"""
import numpy as np
import pandas as pd

NQ = 10
ANN = 242


def tie_broken_decile(scores, date, nq=NQ):
    """确定性随机破并列的十分组. 0=最低分, nq-1=最高分.

    种子用日期数字(非 hash(), 后者受 PYTHONHASHSEED 影响不可复现)。
    """
    rng = np.random.RandomState(int(str(date).replace("-", "")) % (2 ** 31))
    order = np.lexsort((rng.permutation(len(scores)), scores))
    q = np.empty(len(scores), np.int16)
    q[order] = (np.arange(len(scores)) * nq // len(scores))
    return q


def spearman_ic(a, b):
    ar = pd.Series(a).rank()
    br = pd.Series(b).rank()
    return ar.corr(br)


def perf(daily, ann=ANN):
    """日收益序列 → 绩效字典(含单笔指标: 触发数/胜率/单笔均/盈亏比/盈利因子)."""
    x = np.asarray(daily, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {}
    cum = np.cumprod(1 + x)
    dd = 1 - cum / np.maximum.accumulate(cum)
    win, loss = x[x > 0], x[x < 0]
    return {
        "ann_ret": float(x.mean() * ann),
        "ann_sharpe": float(x.mean() / (x.std() + 1e-12) * np.sqrt(ann)),
        "maxDD": float(dd.max()),
        "n_days": int(len(x)),
        "win_rate": float((x > 0).mean()),
        "avg_daily_bp": float(x.mean() * 1e4),
        "payoff": float(win.mean() / abs(loss.mean())) if len(loss) and len(win) else float("nan"),
        "profit_factor": float(win.sum() / abs(loss.sum())) if len(loss) else float("nan"),
    }


# ---------- 标签口径 ----------
# ⚠️ build_dataset.rank_center 把横截面 pct-rank 中心化到 [-0.5, +0.5](不是[0,1])。
# 用「最差20%」这类分位阈值时必须转换, 否则 y<=0.20 会选中最差70%(踩过这个坑)。
def q_thresh(q):
    """[0,1] 分位 q → 中心化秩阈值. q_thresh(0.2)=-0.3 即"最差20%"的上界。"""
    return q - 0.5
