# L2 Microstructure Risk-Guard for A-Share Index Enhancement

**用 247 个 Level-2 微观结构因子做 A 股指数增强的风控规避层。**
每交易日 14:55 对指数成分股预测未来 5 日收益的横截面排名,持有全部成分、
按基准权重、**剔除预测最差的 10%** —— A 股融券受限,空头端只能以"不持有"落地。

> **测试段 2024-07 起,配置于开封前冻结、只评估一次。**
> 中证1000 **超额 +5.11% / IR 3.47 / TE 1.46%**(保留验证段 97%);国证2000 **+4.58% / IR 4.68**(保留 121%)。
> 全部为**双边 20bp 成本后**,基准 = 同域指数权重组合。

---

## 这个仓库最值得看的一点

**同一套管线上,两条线一个通过一个证伪,而且失败是被事前预言的。**

| 线 | 验证段 | 测试段(一次性开封) | 结论 |
|---|---|---|---|
| `riskguard/` 小盘规避层 | +3.89% / IR 3.57 | **+5.11% / IR 3.47 / TE 1.46%** | ✅ 通过,保留 97% |
| `longonly_falsified/` 大盘纯多头选股 | +11.35% / IR 1.53 | **−7.82% / IR −0.74** | ❌ 证伪,如实保留 |

开封**之前**做的风格中性化检验就给出了预警:大盘线 IR 1.53→0.16(超额九成是风格),
小盘规避线 IR 2.96→**3.57**(不降反升,因跟踪误差降得更多)。**同一批因子、同一套管线、
同一个测试窗口,一个垮一个没垮** —— 这排除了"这段行情特殊"的解释。

## 后续复查(2026-08):两把更严的尺子,双双通过

事后用姊妹研究新造的两把标尺回验,**IR 不降反升**:

| | 中证1000 | 国证2000 |
|---|---|---|
| 原始 | +4.93% / IR 2.93 / TE 1.68% | +4.30% / IR 3.56 / TE 1.20% |
| 5 风格(原口径) | +3.87% / IR 3.56 / TE 1.07% | +3.23% / IR 3.89 / TE 0.82% |
| **Barra CNE5 13 风格 + 申万一级行业** | +3.29% / **IR 3.82** / TE 0.84% | +3.31% / **IR 4.80** / TE 0.67% |

**方向是关键**:尺子越严 IR 越高,因为**跟踪误差降得比超额更快**。若超额是风格挣的,
分子会塌得比分母快 —— 大盘那条线正是如此。

**随机剔除零基准**(各 200 次随机剔除 10%)—— 两域都远在零分布之外:

| | 随机剔除(零基准) | 真实剔除 | 超出 | 分位 |
|---|---|---|---|---|
| 中证1000 | **−5.06%**(sd 0.37) | +3.29% | **+8.34pp** | **100%**(约 22σ) |
| 国证2000 | **−6.85%**(sd 0.21) | +3.31% | **+10.16pp** | **100%**(约 49σ) |

随机剔掉 10% 是**亏钱**的(付了换手成本、损失分散度,却没有选择收益);真实剔除是赚的。

## 方法要点

- **无前视**:特征只用当日 09:30–14:55 共 48 根 bar(剔除 15:00 那根);可交易性用 **14:55** 的涨跌停状态;
  purged walk-forward 每 63 交易日重训 + **5 日 embargo**(= 最长标签期,训练集与预测块零重叠)
- **实证核验**:从原始 5 分钟因子库重算日聚合与落盘面板对拍,**ρ(≤14:55 截断)= 1.000000**,
  而 ρ(全部 49 bar)仅 0.988~0.996;另有逐特征 forward-IC 扫描与**对齐证伪测试**(故意错位一天,|IC| 从 0.18 跳到 0.42)
- **成本进目标函数**:`max Σwb·x·α − λ·Σwb·|x−x_prev|` 对个股可分离且线性,
  闭式解是一条 **[−λ, +λ] 无交易带** —— 带宽由成本直接决定,不是调出来的
- **四层稳健性**:移动块自助(21 日块 × 2000 次)为正概率 98~100%;分年全正;配置分布;多种子分布

## 目录

```
src/                    共用底座: 数据构建 / 调度 / 回测工具 / 风格因子
  common.py             路径与常量(全部相对或可用环境变量覆盖)
  build_panel.py        247个5min因子 → 741维日频截面特征
  build_labels.py       前向收益标签 + 14:55 可交易性
  build_dataset.py      域内截面秩变换
  styleutil.py          5 风格(size/vol20/beta20/mom20/rev5)
  barra_styles.py       Barra CNE5 13 风格 + 申万一级行业(PIT 到 d−1)
riskguard/              ★ 交付线: 小盘规避层
  code/                 short_side_exclude.py(主回测) / cost_aware_exclude.py(无交易带)
                        robustness.py / seed_study.py / cost_sensitivity.py / recheck_barra_null.py
  results/              各域结果 + recheck_2026/(两把新尺子的复查)
longonly_falsified/     对照线: 大盘纯多头选股(已被测试段证伪, 如实保留)
legacy_csi500/          前身单域版本, 五条模型层结论的原始证据
docs/
  results_overview.md       完整结果总览
  frozen_test_config.md     ★ 测试段冻结配置(写于开封之前)
  lookahead_audit.md        无前视审计
  handoff.md · checklist.md
data/                   (gitignore) 数据集本地副本
```

## 跑法

**外部数据依赖** —— 本仓**不内嵌任何机器路径**,全部通过环境变量提供(样例见 `.env.example`):

| 环境变量 | 内容 |
|---|---|
| `IDXML_DS_ROOT` | 数据集本地副本(`build_dataset.py` 的产物) |
| `IDXML_FACTOR_ROOT` | 247 个 L2 因子的 5 分钟长表 |
| `IDXML_FWDRET_ROOT` | 1 分钟网格前向收益(构造标签用) |
| `IDXML_LIMIT_ROOT` | 涨跌停/停牌状态(**须含 14:55 时点**) |
| `IDXML_WEIGHT_ROOT` | 指数每日成分与权重 |
| `IDXML_CACHE` | 中间产物缓存(默认 `<repo>/cache`) |

风格因子另需 `IDXML_PRICE_DAILY` / `IDXML_VALUATION` / `IDXML_FININD` / `IDXML_BALANCE` / `IDXML_INDUSTRY`。

```bash
cp .env.example .env      # 按实际填写外部数据路径
set -a && . ./.env && set +a

# 一次性构建(已有 data/ 可跳过)
python src/build_panel.py --index csi1000 gz2000 --workers 60
python src/build_labels.py --index csi1000 gz2000 --workers 40
python src/build_dataset.py --index csi1000

# 训练与回测
python longonly_falsified/code/train_baselines.py --index csi1000 --model lgbm --seeds 3
python riskguard/code/short_side_exclude.py --index csi1000
python riskguard/code/robustness.py --index csi1000
python riskguard/code/recheck_barra_null.py      # Barra 标尺 + 随机剔除零基准
```

## 已知边界

- **测试段已用掉**,任何后续改动都不能再拿它检验
- 容量测算只做到"盈亏平衡点"(实测半价差:csi1000 4.6bp / gz2000 5.8bp),未做完整的冲击成本-规模曲线
- 高 IR 来自**低跟踪误差**而非高收益(超额 5%、TE 1.5%),这限制了规模天花板;
  它的定位是**风控层**,不是 alpha 层
- 大盘域(沪深300/中证500)的纯多头选股线**已证伪**,不要拿来用
