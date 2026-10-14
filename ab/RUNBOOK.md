# Karaoke separation A/B on a short-lived RunPod pod (launch only with David's OK)
Budget: hard $5. Expected ~$0.15-0.35. Key: ~/ClaudeCode/maaravi-analytics/.env RUNPOD_API_KEY (never print).
0. ./prep_inputs.sh                      # done 2026-10-14: work/in/*.wav (צל כבד realigned, checked lag 1 sample)
1. python3 pod.py up --disk 30 --max-minutes 45
2. launchctl submit -l karaoke.ab.watchdog -- /bin/sh -c "cd $PWD && exec $(which python3) pod.py watchdog --max-minutes 50 >> work/watchdog.log 2>&1"
3. (new shell call) pgrep -fl "pod.py watchdog"   # must show it before going on
4. python3 pod.py sshcfg
5. ./run_ab.sh        # setup (self-kill timer armed first), separate hq4,deux,viperx, fetch every 30 s
   if setup >25 min or pod vanishes: pod.py down, then up again (outputs already fetched are kept; separate_ab.py skips done ones)
6. python3 pod.py down   # verifies via API; then: launchctl remove karaoke.ab.watchdog
7. copy work/out + work/in + work/lyr to the box; analyze.py there -> /workspace/karaoke-ab/{clips,results.csv,KEY.json}
