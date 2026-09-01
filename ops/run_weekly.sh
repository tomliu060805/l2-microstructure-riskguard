#!/usr/bin/env bash
# 每周风控规避清单 —— cron 入口
#
# 安装(每周一 17:30 跑, A股收盘后; 信号本身在 14:55 已可得):
#   crontab -e
#   30 17 * * 1 /path/to/repo/ops/run_weekly.sh >> /path/to/repo/ops/logs/cron.log 2>&1
#
# 退出码: 0=正常  3=数据陈旧(清单已产出但不代表当周)  其它=失败
set -u
set -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

# ── 环境 ──
if [ -f "$REPO/.env" ]; then
    set -a; . "$REPO/.env"; set +a
else
    echo "[$(date '+%F %T')] 缺少 $REPO/.env(见 .env.example)" >&2
    exit 1
fi

PY="${IDXML_PYTHON:-/usr/bin/python3}"
THREADS="${IDXML_THREADS:-32}"
TOP="${IDXML_TOP:-200}"

mkdir -p "$REPO/ops/logs" "$REPO/ops/output" "$REPO/ops/state"
STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG="$REPO/ops/logs/weekly_${STAMP}.log"

echo "[$(date '+%F %T')] 开始 (threads=$THREADS top=$TOP)" | tee -a "$LOG"

"$PY" "$REPO/ops/weekly_riskguard.py" --top "$TOP" --threads "$THREADS" "$@" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

case "$rc" in
  0) echo "[$(date '+%F %T')] 完成" | tee -a "$LOG" ;;
  3) echo "[$(date '+%F %T')] 完成, 但数据陈旧 —— 清单不代表当周, 请检查因子库更新" | tee -a "$LOG" ;;
  *) echo "[$(date '+%F %T')] 失败 rc=$rc" | tee -a "$LOG" ;;
esac

# 日志保留 90 天
find "$REPO/ops/logs" -name 'weekly_*.log' -mtime +90 -delete 2>/dev/null
exit "$rc"
