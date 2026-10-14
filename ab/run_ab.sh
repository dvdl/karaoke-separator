#!/bin/bash
# On the Mac, after `pod.py up` + watchdog + `pod.py sshcfg`. Inputs: work/in/*.wav. Outputs: work/out/.
set -eu
cd "$(dirname "$0")"; S="ssh -F work/ssh_cfg pod"; C="scp -F work/ssh_cfg -q"
$C pod_setup.sh separate_ab.py pod:/root/
$S 'cd /root && bash pod_setup.sh' 2>&1 | tee work/setup.log
grep -q SETUP_DONE work/setup.log || { echo "setup failed"; exit 1; }
$S 'mkdir -p /root/in /root/out'; $C work/in/*.wav pod:/root/in/
# run in the background on the pod; fetch whatever is finished every 30 s (pods can vanish)
# pidfile, not pgrep -f: a remote `pgrep -f separate_ab.py` matches its own bash -c command line
$S 'cd /root && (nohup python separate_ab.py --models "${MODELS:-deux,viperx}" --model-dir /models --out /root/out in/*.wav > sep.log 2>&1 & echo $! > sep.pid) && echo started'
mkdir -p work/out
while true; do
  $S 'cd /root/out && find . -name "*.json" | sed "s|^\./||"' 2>/dev/null | while read j; do
    d=$(dirname "$j"); m=$(basename "$j" .json); mkdir -p "work/out/$d"
    [ -f "work/out/$j" ] || { $C "pod:/root/out/$d/${m}_*.flac" "work/out/$d/" && $C "pod:/root/out/$j" "work/out/$j" && echo "fetched $j"; }
  done
  $S 'kill -0 $(cat /root/sep.pid)' 2>/dev/null || break
  sleep 30
done
$C pod:/root/sep.log work/ 2>/dev/null || true
echo "RUN_DONE"; tail -20 work/sep.log
