#!/usr/bin/env bash
src="pagerank-static-vs-dynamic"
out="/home/resources/Documents/subhajit/$src.log"
ulimit -s unlimited
printf "" > "$out"

# Download program
rm -rf $src
git clone https://github.com/puzzlef/$src
cd $src

# Run
g++ -O3 main.cxx
stdbuf --output=L ./a.out ~/data/email-Eu-core-temporal.txt 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/CollegeMsg.txt             2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/sx-mathoverflow.txt        2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/sx-askubuntu.txt           2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/sx-superuser.txt           2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/wiki-talk-temporal.txt     2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/sx-stackoverflow.txt       2>&1 | tee -a "$out"
