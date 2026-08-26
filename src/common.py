# -*- coding: utf-8 -*-
"""L2指增风控 — 公共配置/指数域/日历/切分/测试段封锁.

与前身单域版本(legacy_csi500/)的差异:
  - 单一 CSI500 域 → **四指数参数化**(hs300/csi500/csi1000/csi2000), 四者互斥
  - 面板按四指数**并集**(约3800只)构建一次, 各项目按 index 取子集
  - 缓存目录独立: ${IDXML_CACHE} (不与旧项目混用)

python: /usr/bin/python3 (pandas 1.5.3 / lightgbm 4.7 / xgboost 3.2 / torch-cpu)
"""
import os

import numpy as np
import pandas as pd

# ---------------- 路径 ----------------
# ── 外部数据根 ──────────────────────────────────────────────────────────
# 本仓**不内嵌任何机器路径**。以下五项必须由环境变量提供(见 README「外部数据依赖」)。
# 缺失时给出明确报错, 而不是静默指向一个不存在的目录。
def _req(var, desc):
    v = os.environ.get(var)
    if not v:
        raise EnvironmentError(
            f"缺少环境变量 {var}({desc})。请先按 README 的「外部数据依赖」一节设置。")
    return v


def _opt(var, default):
    return os.environ.get(var, default)


FACTOR_ROOT = _opt("IDXML_FACTOR_ROOT", "")   # 247 个 L2 因子的 5 分钟长表 [datetime,symbol,<factor>]
FWDRET_ROOT = _opt("IDXML_FWDRET_ROOT", "")   # 1min 网格前向收益
LIMIT_ROOT  = _opt("IDXML_LIMIT_ROOT", "")    # 涨跌停 / 停牌状态(需含 14:55 时点)
WEIGHT_ROOT = _opt("IDXML_WEIGHT_ROOT", "")   # 指数每日成分与权重
CACHE       = _opt("IDXML_CACHE", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"))  # 中间产物, 默认落在仓库内

PANEL_DIR = os.path.join(CACHE, "panel_daily")
LABEL_DIR = os.path.join(CACHE, "labels")
WEIGHTS_ALL = os.path.join(CACHE, "index_weights_all.parquet")   # 由 build_weights.py 生成
# 项目根 = 本文件所在目录的上一级(src/ 的父目录)。**不写死绝对路径**, 换机器/改名都不用动。
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------- 指数域 ----------------
# 四个指数按市值分层且**互斥**(实测 2024-01-05 两两交集为0, 并集3500;
# 加沪深300后并集3800)。项目A 四域都做, 项目B 只做 csi1000/csi2000。
INDEXES = {
    "hs300":   {"name": "沪深300",  "dir": "csi_300",  "code": "000300.XSHG", "n": 300},
    "csi500":  {"name": "中证500",  "dir": "csi_500",  "code": "000905.XSHG", "n": 500},
    "csi1000": {"name": "中证1000", "dir": "csi_1000", "code": "000852.XSHG", "n": 1000},
    "csi2000": {"name": "中证2000", "dir": "csi_2000", "code": "932000.CSI",  "n": 2000},
    # gz2000: 重构的"国证2000代理域"(市值秩1001-3000, T-1定域, 权重市值加权封顶0.5%)。
    # 用于替代 csi2000 —— 后者2023年才发布, 2023年前成分为回溯重构(2015年仅310只)。
    # 重构域与官方国证2000(399303.XSHE) 日收益相关 0.9788, 2017年起满2000只。
    # 权重由 build_gz2000_universe.py 单独生成, 不在 index_weights_all.parquet 内。
    "gz2000":  {"name": "国证2000代理", "dir": None, "code": "399303.XSHE", "n": 2000},
}
GZ2000_WEIGHTS = os.path.join(CACHE, "gz2000_weights.parquet")
ALL_INDEXES = [k for k in INDEXES if k != "csi2000"]   # 默认不含 csi2000(已被 gz2000 取代)
RISK_INDEXES = ["csi1000", "gz2000"]      # 项目B 目标域(小盘用重构 gz2000 而非 csi2000)

# ---------------- 窗口与切分 ----------------
DATA_START = "2015-01-05"      # 因子库起点
DATA_END = "2025-12-31"

# 测试段封锁: 该日起为 test, 开发期任何代码不得读取/评估
TEST_START = "2024-07-01"
# 验证段(超参/模型选择用 walk-forward OOS 评估区)
VAL_START = "2022-07-01"

HORIZONS = {"1d": 1, "2d": 2, "5d": 5}
EMBARGO_DAYS = 5

_BAR_1500 = pd.Timestamp("2000-01-01 15:00:00").time()
_BAR_1455 = pd.Timestamp("2000-01-01 14:55:00").time()


def factor_names():
    fs = sorted(os.listdir(FACTOR_ROOT))
    return [f for f in fs if not f.startswith("_")]


def trade_dates():
    """全窗口交易日 (以因子库文件为准)."""
    f0 = factor_names()[0]
    ds = sorted(x[:-8] for x in os.listdir(os.path.join(FACTOR_ROOT, f0)) if x.endswith(".parquet"))
    return [d for d in ds if DATA_START <= d <= DATA_END]


# ---------------- 指数成分 ----------------
def load_weights(indexes=None):
    """长表 [date, index, symbol(6位), weight].
    csi_* 来自 build_weights.py; gz2000 来自 build_gz2000_universe.py。"""
    idxs = indexes or ALL_INDEXES
    parts = []
    if any(i != "gz2000" for i in idxs):
        if not os.path.exists(WEIGHTS_ALL):
            raise FileNotFoundError(f"{WEIGHTS_ALL} 不存在, 先运行 shared/build_weights.py")
        w = pd.read_parquet(WEIGHTS_ALL)
        parts.append(w[w["index"].isin([i for i in idxs if i != "gz2000"])])
    if "gz2000" in idxs:
        if not os.path.exists(GZ2000_WEIGHTS):
            raise FileNotFoundError(f"{GZ2000_WEIGHTS} 不存在, 先运行 shared/build_gz2000_universe.py")
        parts.append(pd.read_parquet(GZ2000_WEIGHTS))
    return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]


def members_by_date(indexes=None):
    """dict: date(str) -> set(6位代码). 多指数取并集; 权重表缺日用前值(ffill)."""
    w = load_weights(indexes)
    grp = {d: set(g) for d, g in w.groupby("date")["symbol"]}
    out, last = {}, None
    for d in trade_dates():
        if d in grp:
            last = grp[d]
        out[d] = last
    return out


def index_weights_by_date(index):
    """dict: date(str) -> DataFrame[symbol, weight] (单指数, 权重已归一)."""
    w = load_weights([index])
    out = {}
    for d, g in w.groupby("date"):
        s = g[["symbol", "weight"]].copy()
        s["weight"] = s["weight"] / s["weight"].sum()
        out[d] = s
    return out


# ---------------- 测试段封锁 ----------------
def is_test_date(d):
    return str(d)[:10] >= TEST_START


def assert_dev_dates(dates, context=""):
    """开发期防窥视: 任何触碰 test 段日期的调用直接抛异常.
    解锁只能通过环境变量 IDXML_TEST_UNLOCK=1 (最终评估一次性使用)."""
    if os.environ.get("IDXML_TEST_UNLOCK") == "1":
        return
    bad = [d for d in dates if is_test_date(d)]
    if bad:
        raise RuntimeError(
            f"[TEST LOCK] {context}: 试图访问测试段日期 {bad[:3]}... (共{len(bad)}天, "
            f"test 自 {TEST_START} 起). 开发期禁止; 最终评估需 IDXML_TEST_UNLOCK=1."
        )


def split_dates():
    """(dev_train, dev_val, test) 三段日期列表."""
    ds = trade_dates()
    tr = [d for d in ds if d < VAL_START]
    va = [d for d in ds if VAL_START <= d < TEST_START]
    te = [d for d in ds if d >= TEST_START]
    return tr, va, te
