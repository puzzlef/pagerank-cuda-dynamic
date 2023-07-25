#!/usr/bin/env bash
src="pagerank-cuda-dynamic-frontier"
ulimit -s unlimited

# Download program
if [[ "$DOWNLOAD" != "0" ]]; then
  rm -rf $src
  git clone https://github.com/puzzlef/$src
fi

# Don't need to download program again.
export DOWNLOAD="0"

# 1. Static vs Dynamic Barrier-free PageRank
export MAX_THREADS="32"
if [[ "$1" == "" || "$1" == "1" ]]; then
  ./"$src/main.sh"
fi

# 2. With strong scaling (fixed batch size)
export BATCH_DELETIONS_BEGIN="0.00005"
export BATCH_DELETIONS_END="0.00005"
export BATCH_INSERTIONS_BEGIN="0.00005"
export BATCH_INSERTIONS_END="0.00005"
export NUM_THREADS_BEGIN="1"
export NUM_THREADS_END="$MAX_THREADS"
export NUM_THREADS_STEP="*=2"
if [[ "$1" == "" || "$1" == "2" ]]; then
  ./"$src/main.sh" "--strong-scaling"
fi

# For uniform failure
export NUM_THREADS_BEGIN="$MAX_THREADS"
export NUM_THREADS_END="$MAX_THREADS"
export FAILURE_THREADS_BEGIN="$MAX_THREADS"
export FAILURE_THREADS_END="$MAX_THREADS"

# 3. With uniform sleep failure
export FAILURE_TYPE="sleep"
if [[ "$1" == "" || "$1" == "3" ]]; then
  ./"$src/main.sh" "--uniform-sleep"
fi

# 4. With uniform crash failure
export FAILURE_TYPE="crash"
if [[ "$1" == "" || "$1" == "4" ]]; then
  ./"$src/main.sh" "--uniform-crash"
fi

# For non-uniform failure
export FAILURE_DURATION_BEGIN="100"
export FAILURE_DURATION_END="100"
export FAILURE_PROBABILITY_BEGIN="0.000001"
export FAILURE_PROBABILITY_END="0.000001"
export FAILURE_THREADS_BEGIN="0"
export FAILURE_THREADS_END="$MAX_THREADS"

# 5. With non-uniform sleep failure
export FAILURE_TYPE="sleep"
if [[ "$1" == "" || "$1" == "5" ]]; then
  ./"$src/main.sh" "--nonuniform-sleep"
fi

# 6. With non-uniform crash failure
export FAILURE_TYPE="crash"
if [[ "$1" == "" || "$1" == "6" ]]; then
  ./"$src/main.sh" "--nonuniform-crash"
fi

# Signal completion
curl -X POST "https://maker.ifttt.com/trigger/puzzlef/with/key/${IFTTT_KEY}?value1=$src"
