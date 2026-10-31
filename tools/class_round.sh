#!/bin/sh
# One virtual class round (10-31): stop Coal's app (every copy on the port), run the virtual
# class, start the app again and put the live question back up.
cd "$(dirname "$0")/.." || exit 1
PY="$LOCALAPPDATA/Programs/Python/Python312/python.exe"
Q=$(curl -s localhost:8765/api/class/live | "$PY" -c "import sys,json;print(json.load(sys.stdin).get('question',''))" 2>/dev/null)
powershell -NoProfile -c "Get-NetTCPConnection -LocalPort 8765 -State Listen -EA SilentlyContinue | % { Stop-Process -Id \$_.OwningProcess -Force }"
PYTHONIOENCODING=utf-8 "$PY" -m household.auto_class
(nohup "$PY" -m household.app > logs/household_app.log 2>&1 &)
sleep 8
[ -n "$Q" ] && "$PY" -c "import sys,json,urllib.request as u;u.urlopen(u.Request('http://localhost:8765/api/class/live/ask',json.dumps({'question':sys.argv[1]}).encode(),{'Content-Type':'application/json'}))" "$Q"
"$PY" -c "import torch;k=torch.load('checkpoints/household_brain_trend.pt')['keep'];print('answers',len(k['answers']['texts']),'typed words',len(k.get('type_words',[])))"
