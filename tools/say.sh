#!/bin/sh
# Say one line to Coal in the running class session (tools/class_session.py) and print his reply.
D="$(dirname "$0")/../logs/class"
mv -f "$D/out.txt" "$D/out.prev" 2>/dev/null
printf '%s\n' "$*" > "$D/in.tmp" && mv "$D/in.tmp" "$D/in.txt"
i=0
while [ ! -f "$D/out.txt" ] && [ $i -lt 1200 ]; do sleep 0.25; i=$((i+1)); done
cat "$D/out.txt" 2>/dev/null || echo "[no reply in 5 min]"
