# -*- coding: utf-8 -*-
"""给首跑产出的模型状态补上因子源指纹(一次性)。

首跑时进程已加载了加固前的模块, 存下的 state 里没有 factor_source 字段;
不补的话下次运行会判定"旧状态缺指纹"而整体重训一次, 白烧一小时。
这些模型确实是在 v1 上训的, 所以按当前 FACTOR_ROOT 补指纹是如实记录, 不是绕过校验。
"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
import common as C
from weekly_riskguard import factor_source_fingerprint, STATE_DIR

fp = factor_source_fingerprint()
assert fp["root"].rstrip("/").endswith("factor_store/5m"), \
    f"当前 FACTOR_ROOT 不是 v1({fp['root']}), 拒绝补写 —— 指纹必须如实反映训练时的源"
print(f"指纹: {fp}")
for f in sorted(os.listdir(STATE_DIR)):
    if not f.endswith(".pkl"):
        continue
    p = os.path.join(STATE_DIR, f)
    with open(p, "rb") as fh:
        st = pickle.load(fh)
    if st.get("factor_source"):
        print(f"  {f}: 已有指纹, 跳过"); continue
    st["factor_source"] = fp
    with open(p, "wb") as fh:
        pickle.dump(st, fh)
    print(f"  {f}: 已补 (fit_date={st['fit_date']})")
