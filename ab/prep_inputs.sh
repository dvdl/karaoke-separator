#!/bin/bash
# On the Mac (YouTube refuses the server). Downloads the test songs' audio, 44.1 kHz stereo WAV.
set -eu
cd "$(dirname "$0")"; mkdir -p work/in
Y=~/.local/share/karaoke-fetcher/venv/bin/yt-dlp
for id in Oj_PB40UC84 0tFm6qBLsEA E39srf6DWU8 wJ7p1GMkmkM dQw4w9WgXcQ; do
  [ -f work/in/$id.wav ] && continue
  for t in 1 2 3; do $Y -q --no-warnings -f bestaudio -o "work/$id.%(ext)s" "https://www.youtube.com/watch?v=$id" && break; sleep 5; done
  f=$(ls work/$id.* | grep -v wav | head -1)
  if [ $id = Oj_PB40UC84 ]; then
    # צל כבד: channels 10.0 ms apart (CLAUDE.md); measured: the LEFT lags, so delay the right
    ffmpeg -loglevel error -y -i "$f" -af "adelay=0|10" -ar 44100 -ac 2 work/in/$id.wav
  else
    ffmpeg -loglevel error -y -i "$f" -ar 44100 -ac 2 work/in/$id.wav
  fi
  rm -f "$f"
done
ls -la work/in
