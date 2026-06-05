#!/usr/bin/env bash
# Standalone, patient retry for ONLY the candidate parent (test-site) embeddings,
# which the DoR/CI/GHM sweep (buffer_extent_sweep --candidate) needs. The loss
# parents are not in the v4 stable test-site file. EE is in restricted mode and
# 429s at session init, so retry with long backoff. On success, run the DoR/CI
# sweep with --candidate and rebuild the report (now all 5 classes in every axis).
#
# Detach:  setsid nohup bash v5/scripts/extraction/retry_candidate_parents.sh \
#             > /tmp/candidate_parents.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
REFS="v5/data/v5_candidate_refs_alphaearth.parquet"
TS="v5/data/test_site_alphaearth_2024_candidate.parquet"

MAX_TRIES="${MAX_TRIES:-40}"      # ~ up to 40 * 5min = 3.3h of patient retrying
GAP="${GAP:-300}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

if [ ! -f "$REFS" ]; then log "[abort] candidate refs missing: $REFS"; exit 1; fi

n=0
while [ ! -f "$TS" ] && [ "$n" -lt "$MAX_TRIES" ]; do
  n=$((n + 1))
  log "parent-extract attempt $n/$MAX_TRIES"
  python3 -u v1/scripts/extraction/extract_test_site_embeddings.py \
    --refs "$REFS" --year 2024 --output "$TS" 2>&1 | sed 's/^/    /'
  if [ -f "$TS" ]; then log "parent embeddings extracted on attempt $n"; break; fi
  log "attempt $n failed (likely EE 429); sleeping ${GAP}s"
  sleep "$GAP"
done

if [ ! -f "$TS" ]; then
  log "[give up] parent extraction did not succeed after $MAX_TRIES tries; re-run later."
  exit 1
fi

log "running DoR/CI/GHM sweep with --candidate ..."
python3 v5/scripts/analysis/buffer_extent_sweep.py --candidate 2>&1 | sed 's/^/    /'

log "rebuilding desirability hypercube + report (all 5 classes) ..."
python3 v5/scripts/analysis/buffer_desirability.py 2>&1 | sed 's/^/    /'

log "[done] report rebuilt with loss sites in every axis: v5/report/buffer_decision.html"
