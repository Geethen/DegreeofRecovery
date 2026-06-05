#!/usr/bin/env bash
# Finish the built_loss/crop_loss arm of the buffer decision once the candidate
# REFERENCE embeddings have been extracted (by retry_candidate_extraction.sh).
#
# Steps:
#   1. ensure candidate PARENT (test-site) embeddings exist (small extraction;
#      the loss parents are NOT in the stable v4 test-site file).
#   2. re-run the three sweeps with --candidate so built_loss/crop_loss enter the
#      separability, DoR/CI/GHM, and spatial-autocorrelation axes.
#   3. (optional) refresh the contamination input from the v5 candidate refs.
#   4. rebuild the desirability hypercube + HTML report.
#
# Run:  bash v5/scripts/analysis/run_buffer_pipeline_with_candidate.sh
set -eu
cd "$(dirname "$0")/../../.."
REPO="$(pwd)"

CAND_REFS="v5/data/v5_candidate_refs_alphaearth.parquet"
CAND_TS="v5/data/test_site_alphaearth_2024_candidate.parquet"

if [ ! -f "$CAND_REFS" ]; then
  echo "[abort] candidate ref parquet not found: $CAND_REFS"
  echo "        run/await v5/scripts/extraction/retry_candidate_extraction.sh first."
  exit 1
fi
echo "[ok] candidate ref parquet: $CAND_REFS"
python3 -c "import duckdb;print('     ref rows:', duckdb.connect().execute(\"SELECT count(*) FROM '$CAND_REFS'\").fetchone()[0])"

# --- 1. candidate parent (test-site) embeddings -----------------------------
if [ -f "$CAND_TS" ]; then
  echo "[ok] candidate parent embeddings present: $CAND_TS"
else
  echo "[extract] candidate parent (test-site) embeddings -> $CAND_TS"
  n=0
  until [ -f "$CAND_TS" ] || [ "$n" -ge 8 ]; do
    n=$((n + 1))
    python3 -u v1/scripts/extraction/extract_test_site_embeddings.py \
      --refs "$CAND_REFS" --year 2024 --output "$CAND_TS" || true
    [ -f "$CAND_TS" ] && break
    echo "  parent-extract attempt $n failed (likely EE 429); sleeping 300s"
    sleep 300
  done
  [ -f "$CAND_TS" ] || { echo "[abort] could not extract candidate parents after retries"; exit 1; }
fi

# --- 2. sweeps with candidate ----------------------------------------------
echo ""
echo "=== separability incl. candidate ==="
python3 v5/scripts/analysis/separability_sweep.py --candidate

echo ""
echo "=== buffer extent (DoR / CI / GHM corr) incl. candidate ==="
python3 v5/scripts/analysis/buffer_extent_sweep.py --candidate

echo ""
echo "=== spatial autocorrelation incl. candidate ==="
python3 v5/scripts/analysis/spatial_autocorr_sweep.py --candidate

# --- 3. rebuild hypercube + report -----------------------------------------
echo ""
echo "=== rebuild desirability hypercube + report ==="
python3 v5/scripts/analysis/buffer_desirability.py

echo ""
echo "[done] report: v5/report/buffer_decision.html  (now includes built_loss/crop_loss)"
