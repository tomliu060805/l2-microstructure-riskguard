# LGBM 特征重要性稳定性与归因 (gain, 3种子均值, 25个refit块)

## 1. 时间稳定性

- 相邻块重要性 Spearman 秩相关: 均值 **0.841** (范围 0.688~0.912)
- 首块 vs 末块(2018 vs 2024): **0.768**
- top50 相邻块重合率: 均值 **85%**
- 全部 25 个块都进 top50 的特征: **21 个**

解读: 秩相关高=模型学到的结构跨市场状态稳定(而非每次refit都换一批特征), 这是信号可信的必要条件; 若接近0则说明纯拟合噪声。

## 2. 日内聚合类型的重要性份额

| 聚合 | 含义 | 特征数 | 重要性份额 |
|---|---|---|---|
| mean | 全日均值 | 247 | 37.1% |
| std | 日内波动 | 247 | 33.7% |
| lh | 尾盘14:00-14:55均值 | 247 | 29.0% |
| other | coverage | 1 | 0.2% |

## 3. 因子级 top20 (三个聚合的gain求和)

| # | 因子 | 重要性份额 |
|---|---|---|
| 1 | `trade_size_tail_5m` | 6.46% |
| 2 | `near_uplimit_buy_press_5m` | 3.27% |
| 3 | `child_size_quantization_5m` | 3.04% |
| 4 | `order_price_range_pos_mean_5m` | 2.88% |
| 5 | `order_price_range_skew_5m` | 2.71% |
| 6 | `exploratory_spread_width_5m` | 2.10% |
| 7 | `order_lifetime_tail_5m` | 1.96% |
| 8 | `downlimit_buy_support_5m` | 1.90% |
| 9 | `deindividuation_5m` | 1.82% |
| 10 | `disposition_effect_proxy_5m` | 1.75% |
| 11 | `corr_length_5m` | 1.63% |
| 12 | `humble_buildup_vs_display_5m` | 1.49% |
| 13 | `impact_perm_side_asym_5m` | 1.42% |
| 14 | `attn_sync_swarm_5m` | 1.25% |
| 15 | `overpay_regret_mag_5m` | 1.25% |
| 16 | `downlimit_capitulation_vs_defense_5m` | 1.18% |
| 17 | `sticker_anchor_entry_gate_5m` | 1.17% |
| 18 | `support_defense_count_5m` | 1.13% |
| 19 | `buy_sell_order_size_asym_5m` | 1.02% |
| 20 | `queue_jump_abandonment_5m` | 1.02% |

- 头部集中度: top20因子占 **40.4%**, top50占 **61.7%** (共248个因子)
