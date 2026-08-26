# 对接文档 — 接口约定与数据契约

本文件规定三类接口：**共享层 ↔ 项目**、**项目A ↔ 项目B**、**平台 ↔ 外部系统**。
任何一方改动接口都必须先改本文件。

---

## 一、共享层 ↔ 项目

### 1.1 数据契约（各环节的输入输出）

| 环节 | 脚本 | 输入 | 输出 | 关键约定 |
|---|---|---|---|---|
| 权重 | `build_weights.py` | `${IDXML_WEIGHT_ROOT}/{csi_300,csi_500,csi_1000,csi_2000}/<date>.parquet` | `{CACHE}/index_weights_all.parquet`<br>`[date, index, symbol(6位), weight]` | symbol 一律 **6位数字**（去交易所后缀） |
| 面板 | `build_panel.py` | `{FACTOR_ROOT}/<factor>/<date>.parquet` | `{CACHE}/panel_daily/<date>.parquet`<br>`[symbol, coverage, <factor>\|mean/std/lh × 247]` | 域=四指数**并集**；只用 09:30–14:55（**剔除 15:00 bar**） |
| 标签 | `build_labels.py` | `{FWDRET_ROOT}/<date>.parquet`<br>`{LIMIT_ROOT}/<date>.parquet` | `{CACHE}/labels/<date>.parquet`<br>`[symbol, ret_fwd_1d/2d/5d, tradable]` | `tradable` 用 **14:55** 的 limit/paused 判定 |
| 数据集 | `build_dataset.py --index X` | 面板 + 标签 + 权重 | `{CACHE}/dataset/<index>/ds.parquet`<br>`[date, symbol, weight, tradable, ret_fwd_*, y1d_rank, y5d_rank, <741特征>]` | **秩变换在该指数当日成分内计算** |

`CACHE = ${IDXML_CACHE}`

### 1.2 为什么数据集按指数分开

四个域各自独立建模，模型运行时看到的截面就是该指数当日成分。若在并集（3800只）上做
横截面秩，中证2000 个股的秩会被沪深300 大盘股拉偏，**与模型实际运行的截面不一致**。
因此秩变换在各指数域内计算，每个指数产出独立数据集。

**代价**：同一只票在不同指数的数据集里秩不同——但四指数互斥，同一票同一天只属于一个域，
不存在冲突。

### 1.3 关键数值约定（踩过坑，务必遵守）

| 约定 | 说明 |
|---|---|
| **秩是中心化的 `[-0.5, +0.5]`，不是 `[0,1]`** | 用分位阈值必须转换：用 `btutil.q_thresh(0.2) = -0.3` 表示"最差20%"。直接写 `y <= 0.20` 会选中最差 **70%**（已在旧项目造成过错误结论） |
| 破并列用日期数字做种子 | L2 因子大量取值并列，按秩分组会把整块并列压进同一组造出假收益。用 `btutil.tie_broken_decile`，**不可用 `hash()`**（受 `PYTHONHASHSEED` 影响不可复现） |
| 年化天数 `ANN = 242` | 全平台统一 |
| NaN 保留不填充 | LGBM 原生处理；线性基线内部填 0（=秩的中位） |

### 1.4 调用接口

```python
import sys; sys.path.insert(0, "<平台根>/shared")
import common as C
from dsutil import load_ds, schedule, ds_dir, pred_dir, split_train_es
from btutil import tie_broken_decile, spearman_ic, perf, q_thresh, ANN

ds, feats = load_ds("csi1000")          # 按指数加载
sched = schedule(sorted(ds["date"].unique()))   # [(train_end_idx, [predict dates])]
C.assert_dev_dates(dates, "上下文")      # 触碰测试段直接抛异常
w = C.index_weights_by_date("csi1000")  # {date -> DataFrame[symbol, weight]}
```

---

## 二、测试段防火墙（强制）

- **封锁日**：`TEST_START = 2024-07-01`
- **机制**：`common.assert_dev_dates(dates, context)` 检测到任何 ≥ 该日的日期即抛 `RuntimeError`
- **解锁**：仅 `IDXML_TEST_UNLOCK=1`，且约定**只跑一次，结果不得回头改模型**
- **例外**：面板/标签/数据集**构建**可覆盖全窗口（构建 ≠ 评估）；训练与回测侧必须过检查

预测文件在解锁模式下自动加 `_WITHTEST` 后缀，避免与开发期结果混淆。

---

## 三、项目A ↔ 项目B

两个项目**独立训练、独立结论**，但共享同一数据集与回测原语，因此结果可比、可叠加。

### 3.1 边界

| | 项目A 纯多头选股 | 项目B 指增风控规避 |
|---|---|---|
| 目标函数 | 对称回归（`y5d_rank`） | 下尾专用（二分类 / 分位数回归 q=0.10） |
| 关心的截面位置 | **上尾**（谁会涨） | **下尾**（谁会跌） |
| 输出用途 | 选出持仓 | 选出**不持有**的票 |
| 预测文件 | `{CACHE}/preds/<index>/lgbm_y5d_rank.parquet` | `{CACHE}/preds/<index>/tail_<obj>_y5d.parquet` |
| 域 | hs300 / csi500 / csi1000 / csi2000 | csi1000 / csi2000 |

### 3.2 预测文件统一格式（两项目一致）

```
[date(str, YYYY-MM-DD), symbol(6位), score(float32)]
```
`score` 越大越看好。项目B 的下尾模型输出也遵循此约定（内部已翻转符号），
使得**同一个回测器可以直接吃两个项目的分数**，差异只来自信号本身。

### 3.3 叠加方式（组合成完整指增产品）

在同一指数域上：

```
基准权重（指数权重）
  → 项目B 规避层：剔除下尾模型预测最差 K%（5日信号平滑）
  → 项目A alpha 层：在剩余票中按选股分数超配/低配
  → 目标权重
```

**注意顺序**：先规避后选股。反过来会让 alpha 层把资金配到随后被规避层剔除的票上，
产生无意义换手。

**叠加效果必须单独回测**，不可用两者单独结果相加——两个信号相关性未知，
叠加后换手也会变化。这是尚未完成的工作（见各项目 README 的待办）。

---

## 四、平台 ↔ 外部系统（实盘对接）

### 4.1 每日运行时序

| 时点 | 动作 | 责任方 |
|---|---|---|
| 09:30–14:55 | L2 5min 因子累积 | 上游因子链路 |
| 14:55 | 日频聚合 → 741 特征 → 模型推断 → 5日信号平滑 | 本平台 |
| 14:55–15:00 | 生成目标权重 → 下单 | 交易系统 |
| 次日 15:00 | 标签口径起算（`ret_fwd_1d` = 次日收/今收−1） | — |

**最硬的约束：决策与下单窗口只有 5 分钟。** 若实盘只能次日开盘执行，
**全部回测结论作废，必须重跑**——尾盘信号衰减很快。

### 4.2 交付给交易系统的接口

建议格式（每日一个文件）：

```
target_weights_<index>_<date>.csv
  symbol, target_weight, benchmark_weight, action(hold/exclude), score, reason
```

`reason` 字段建议保留：`tail_excluded` / `not_tradable_keep` / `alpha_overweight` 等，
便于事后归因与合规审计。

### 4.3 不可交易票的处理约定

决策时点（14:55）封涨跌停或停牌的票 → **保留基准权重，不强行剔除**。
这是回测中的既定口径（`tradable=0` 的票不参与剔除），实盘必须一致，否则回测失真。

---

## 五、环境与运维

| 项 | 值 |
|---|---|
| Python | `/usr/bin/python3`（pandas 1.5.3 / lightgbm 4.7 / xgboost 3.2 / torch-cpu） |
| 大缓存 | `${IDXML_CACHE}/` |
| 本地副本 | 训练前把 `dataset/<index>/` 复制到本地盘，设 `IDXML_DS_ROOT` 指向它 |
| 测试段解锁 | `IDXML_TEST_UNLOCK=1`（一次性） |

**已知运维坑**：
1. 共享数据盘长期 I/O 饱和（util 97%、延迟 600ms）。不设 `IDXML_DS_ROOT` 而直接读共享盘
   可能让单次数据集加载超过 1 小时
2. `pgrep -f "xxx"` 会匹配到自身的 wrapper 命令行，导致死循环或误杀（`pkill` 同理）。
   用 PID 或加排除条件
3. 后台长任务用 `setsid nohup ... &` 启动，否则终端关闭时可能被杀

---

## 六、变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-07 | 平台建立。从单域 CSI500 版本(见 `legacy_csi500/`)扩展为四指数双项目结构 |
