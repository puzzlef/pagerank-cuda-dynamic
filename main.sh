#!/usr/bin/env bash
src="pagerank-cuda-dynamic-frontier"
out="$HOME/Logs/$src$1.log"
ulimit -s unlimited
printf "" > "$out"

# Download program
if [[ "$DOWNLOAD" != "0" ]]; then
  rm -rf $src
  git clone https://github.com/puzzlef/$src
fi
cd $src

# Fixed config
: "${TYPE:=double}"
: "${MAX_THREADS:=32}"
: "${REPEAT_BATCH:=5}"
: "${REPEAT_METHOD:=1}"
# Parameter sweep for batch (randomly generated)
: "${BATCH_DELETIONS_BEGIN:=0.000000005}"
: "${BATCH_DELETIONS_END:=0.05}"
: "${BATCH_DELETIONS_STEP:=*=10}"
: "${BATCH_INSERTIONS_BEGIN:=0.000000005}"
: "${BATCH_INSERTIONS_END:=0.05}"
: "${BATCH_INSERTIONS_STEP:=*=10}"
# Define macros (dont forget to add here)
DEFINES=(""
"-DTYPE=$TYPE"
"-DMAX_THREADS=$MAX_THREADS"
"-DREPEAT_BATCH=$REPEAT_BATCH"
"-DREPEAT_METHOD=$REPEAT_METHOD"
"-DBATCH_DELETIONS_BEGIN=$BATCH_DELETIONS_BEGIN"
"-DBATCH_DELETIONS_END=$BATCH_DELETIONS_END"
"-DBATCH_DELETIONS_STEP=$BATCH_DELETIONS_STEP"
"-DBATCH_INSERTIONS_BEGIN=$BATCH_INSERTIONS_BEGIN"
"-DBATCH_INSERTIONS_END=$BATCH_INSERTIONS_END"
"-DBATCH_INSERTIONS_STEP=$BATCH_INSERTIONS_STEP"
)

# Run
g++ ${DEFINES[*]} -std=c++17 -O3 -fopenmp main.cxx -o "a$1.out"
# stdbuf --output=L ./"a$1.out" ~/Data/soc-Epinions1.mtx  2>&1 | tee -a "$out"
stdbuf --output=L ./"a$1.out" ~/Data/indochina-2004.mtx  2>&1 | tee -a "$out"
stdbuf --output=L ./"a$1.out" ~/Data/uk-2002.mtx         2>&1 | tee -a "$out"
stdbuf --output=L ./"a$1.out" ~/Data/arabic-2005.mtx     2>&1 | tee -a "$out"
stdbuf --output=L ./"a$1.out" ~/Data/uk-2005.mtx         2>&1 | tee -a "$out"
stdbuf --output=L ./"a$1.out" ~/Data/webbase-2001.mtx    2>&1 | tee -a "$out"
stdbuf --output=L ./"a$1.out" ~/Data/it-2004.mtx         2>&1 | tee -a "$out"
stdbuf --output=L ./"a$1.out" ~/Data/sk-2005.mtx         2>&1 | tee -a "$out"
stdbuf --output=L ./"a$1.out" ~/Data/com-LiveJournal.mtx 2>&1 | tee -a "$out"
stdbuf --output=L ./"a$1.out" ~/Data/com-Orkut.mtx       2>&1 | tee -a "$out"
stdbuf --output=L ./"a$1.out" ~/Data/asia_osm.mtx        2>&1 | tee -a "$out"
stdbuf --output=L ./"a$1.out" ~/Data/europe_osm.mtx      2>&1 | tee -a "$out"
stdbuf --output=L ./"a$1.out" ~/Data/kmer_A2a.mtx        2>&1 | tee -a "$out"
stdbuf --output=L ./"a$1.out" ~/Data/kmer_V1r.mtx        2>&1 | tee -a "$out"

# Signal completion
curl -X POST "https://maker.ifttt.com/trigger/puzzlef/with/key/${IFTTT_KEY}?value1=$src$1"
