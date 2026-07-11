#!/bin/bash
# Fetch Metrica Sports open tracking data (not redistributed with this repo).
# Source: https://github.com/metrica-sports/sample-data
set -euo pipefail
DIR="$(dirname "$0")/../data"
mkdir -p "$DIR"
BASE=https://raw.githubusercontent.com/metrica-sports/sample-data/master/data
for g in 1 2; do
  for side in Home Away; do
    f="Sample_Game_${g}_RawTrackingData_${side}_Team.csv"
    [ -f "$DIR/$f" ] || curl -sL -o "$DIR/$f" "$BASE/Sample_Game_$g/$f"
    echo "ok $f"
  done
done
