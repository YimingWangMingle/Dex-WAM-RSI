#!/usr/bin/env bash
# 监控前端 JS 语法校验（零依赖，需 node）：对 web/js 下全部 ES module 做 parse 检查。
# 用法：bash libs/motus-monitor/tools/check_js.sh
set -u
cd "$(dirname "$0")/.."

fail=0
count=0
while IFS= read -r f; do
  count=$((count + 1))
  if ! node --input-type=module --check < "$f" 2>/tmp/motus_jscheck_err; then
    echo "FAIL $f"
    sed 's/^/  /' /tmp/motus_jscheck_err | head -8
    fail=1
  fi
done < <(find motus_monitor/web/js -name '*.js' 2>/dev/null | sort)

if [ "$count" -eq 0 ]; then
  echo "no files found under motus_monitor/web/js (nothing to check)"
  exit 0
fi
if [ "$fail" -eq 0 ]; then
  echo "OK: $count file(s) parsed as ES modules"
else
  echo "FAILED"
fi
exit "$fail"
