#!/usr/bin/env bash

set -e
set -o pipefail

# Run from the repo root so the relative paths below resolve regardless of
# where the script is invoked from.
cd "$(dirname "$0")"

# --- adjust for your box if needed ---
PYTHON="${PYTHON:-python3}"
ENGINE="./engines/deuterium/Deuterium_v2019_linux_64"   # Linux; chmod +x it
# -------------------------------------

# Use a local virtualenv if one is present.
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    PYTHON="python"
fi

# Plots are saved here as interactive .html files.
mkdir -p visuals

"$PYTHON" tuner.py \
  --study-name deuterium_study_1_tpe \
  --sampler name=tpe \
  --engine "$ENGINE" \
  --concurrency 2 \
  --opening-file ./start_opening/ogpt_chess_startpos.epd \
  --opening-format epd \
  --input-param "{'RazorMargin1': {'default':220, 'min':150, 'max':400, 'step':1}, 'QSearchFutilityMargin': {'default':100, 'min':50, 'max':200, 'step':1}, 'ProbCutMargin': {'default':200, 'min':100, 'max':350, 'step':1}, 'FutilityMargin': {'default':60, 'min':25, 'max':300, 'step':1}, 'MobilityWeight': {'default':100, 'min':10, 'max':1000, 'step':2}}" \
  --games-per-trial 50 \
  --trials 20 \
  --plot \
  --draw-movenumber 35 --draw-movecount 6 --draw-score 0 \
  --resign-movecount 3 --resign-score 700 \
  --base-time-sec 10 --inc-time-sec 0.1 \
  --pgn-output out_deuterium_study_1_tpe.pgn \
  --threshold-pruner result=0.35
