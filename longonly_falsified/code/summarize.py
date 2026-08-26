# -*- coding: utf-8 -*-
"""一键汇总: 对所有预测跑统一回测(多空tranche5 + 多头超额), 输出总表 markdown.

用法: /usr/bin/python3 summarize.py [--preds a b c] [--window val]
"""
import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
RESULTS = os.path.join(HERE, "..", "results")
import common as C
from backtest import run

LABELS = {
    "linear_y5d_rank": ("IC加权线性(top60因子)", "全部741"),
    "lgbm_y5d_rank": ("LightGBM (3种子)", "全部741"),
    "lgbm_y5d_rank_imp": ("LightGBM 复跑(重要性版)", "全部741"),
    "xgb_y5d_rank": ("XGBoost (3种子)", "全部741"),
    "ens_lgbm_xgb": ("LGBM+XGB 秩集成", "全部741"),
    "nnmlp_y5d_rank": ("MLP 512-128 (2种子)", "全部741"),
    "nngru_y5d_rank": ("GRU-96, 5日序列 (2种子)", "247×5天"),
    "lgbm_y5d_rank_agg_mean": ("LGBM 消融: 仅日均值", "247 (mean)"),
    "lgbm_y5d_rank_agg_meanstd": ("LGBM 消融: 均值+波动", "494 (mean,std)"),
    "lgbm_y5d_rank_top50": ("LGBM 消融: top50特征", "50"),
}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--index", required=True,

                    help="hs300/csi500/csi1000/csi2000 — 每个指数域独立建模")
    ap.add_argument("--preds", nargs="*", default=None)
    ap.add_argument("--window", default="val")
    args = ap.parse_args()

    pred_dir = os.path.join(C.CACHE, "preds")
    avail = [f[:-8] for f in sorted(os.listdir(pred_dir))
             if f.endswith(".parquet") and not f.startswith("importance_")]
    preds = args.preds or [p for p in LABELS if p in avail]

    rows_ls, rows_long = [], []
    for p in preds:
        r = run(p, args.window, unlock=False, style="tranche5", side="ls")
        rl = run(p, args.window, unlock=False, style="tranche5", side="long")
        name, fs = LABELS.get(p, (p, "?"))
        rows_ls.append({
            "模型": name, "特征集": fs,
            "RankIC(5d)": r["rank_ic_5d"]["mean"], "日ICIR": r["rank_ic_5d"]["icir_daily"],
            "IC正天占比": r["rank_ic_5d"]["pct_pos"],
            "D1年化": r["decile_ann_ret"]["D1"], "D10年化": r["decile_ann_ret"]["D10"],
            "单边日换手": r["turnover_oneway_daily"],
            "毛Sharpe": r["LS_gross"]["ann_sharpe"],
            "净20bp Sharpe": r["LS_net_20bp"]["ann_sharpe"],
            "净30bp Sharpe": r["LS_net_30bp"]["ann_sharpe"],
            "净20bp年化": r["LS_net_20bp"]["ann_ret"],
            "净20bp胜率": r["LS_net_20bp"]["win_rate"],
            "净20bp盈亏比": r["LS_net_20bp"]["payoff"],
            "净20bp盈利因子": r["LS_net_20bp"]["profit_factor"],
            "交易日数": r["LS_net_20bp"]["n_days"],
            "maxDD(净20bp)": r["LS_net_20bp"]["maxDD"],
        })
        rows_long.append({
            "模型": name,
            "组合年化": rl["port_ann_ret"], "基准年化": rl["bench_ann_ret"],
            "超额毛": rl["excess_gross"]["ann_ret"], "毛IR": rl["excess_gross"]["info_ratio"],
            "TE": rl["excess_gross"]["tracking_error"],
            "超额净20bp": rl["excess_net_20bp"]["ann_ret"],
            "净20bp IR": rl["excess_net_20bp"]["info_ratio"],
            "超额净30bp": rl["excess_net_30bp"]["ann_ret"],
            "净30bp IR": rl["excess_net_30bp"]["info_ratio"],
            "单边日换手": rl["turnover_oneway_daily"],
            "超额胜率": rl["excess_net_20bp"]["win_rate"],
            "超额盈利因子": rl["excess_net_20bp"]["profit_factor"],
        })
        print("done", p, flush=True)

    ls = pd.DataFrame(rows_ls)
    lo = pd.DataFrame(rows_long)
    out_dir = os.path.join(RESULTS)
    ls.to_csv(os.path.join(out_dir, f"总表_多空_{args.window}.csv"), index=False)
    lo.to_csv(os.path.join(out_dir, f"总表_多头超额_{args.window}.csv"), index=False)
    with open(os.path.join(out_dir, f"总表_{args.window}.json"), "w") as f:
        json.dump({"ls": rows_ls, "long": rows_long}, f, ensure_ascii=False, indent=2)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(ls.round(3).to_string(index=False))
    print()
    print(lo.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
