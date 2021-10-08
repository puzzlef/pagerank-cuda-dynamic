# https://www.kaggle.com/wolfram77/puzzlef-pagerank-cuda-static-vs-incremental
import os
from IPython.display import FileLink
src="pagerank-cuda-static-vs-incremental"
inp="/kaggle/input/graphs"
out="{}.txt".format(src)
!printf "" > "$out"
display(FileLink(out))
!ulimit -s unlimited && echo ""
!nvidia-smi && echo ""

# Download program
!rm -rf $src
!git clone https://github.com/puzzlef/$src
!echo ""

# Run
!nvcc -std=c++17 -Xcompiler -DNVGRAPH_DISABLE -O3 $src/main.cu
!stdbuf --output=L ./a.out $inp/email-Eu-core-temporal.txt 2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/CollegeMsg.txt             2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/sx-mathoverflow.txt        2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/sx-askubuntu.txt           2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/sx-superuser.txt           2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/wiki-talk-temporal.txt     2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/sx-stackoverflow.txt       2>&1 | tee -a "$out"
