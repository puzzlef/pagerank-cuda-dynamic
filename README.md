Performance of static vs dynamic CUDA based PageRank ([pull], [CSR], [scaled-fill]).

This experiment was for comparing the performance between:
1. Find **static** pagerank of updated graph.
2. Find **dynamic** pagerank of updated graph.

Both techniques were attempted on different temporal graphs, updating each
graph with multiple batch sizes (`1`, `5`, `10`, `50`, ...). New edges are
incrementally added to the graph batch-by-batch until the entire graph is
complete. **Dynamic** pagerank is clearly **faster** that *static* approach
for many batch sizes.

All outputs are saved in [out](out/) and a small part of the output is listed
here. Some [charts] are also included below, generated from [sheets]. The input
data used for this experiment is available at the
[Stanford Large Network Dataset Collection].

<br>

```bash
$ nvcc -std=c++17 -Xcompiler -lnvgraph -O3 main.cu
$ ./a.out ~/data/email-Eu-core-temporal.txt
$ ./a.out ~/data/CollegeMsg.txt
$ ...

# ...
#
# Using graph sx-stackoverflow ...
# Temporal edges: 63497051
# order: 2601977 size: 36233450 {}
#
# # Batch size 1e+0
# [00050.751 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# [00041.114 ms; 054 iters.] [1.1043e-6 err.] pagerankCuda [static]
# [00002.870 ms; 000 iters.] [1.2991e-6 err.] pagerankNvgraph [dynamic]
# [00000.907 ms; 001 iters.] [1.1864e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 5e+0
# [00050.715 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# [00041.075 ms; 054 iters.] [1.1059e-6 err.] pagerankCuda [static]
# [00006.075 ms; 000 iters.] [2.1583e-6 err.] pagerankNvgraph [dynamic]
# [00001.208 ms; 002 iters.] [1.4482e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 1e+1
# [00050.815 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# [00041.122 ms; 054 iters.] [1.1299e-6 err.] pagerankCuda [static]
# [00008.783 ms; 000 iters.] [2.6527e-6 err.] pagerankNvgraph [dynamic]
# [00001.584 ms; 003 iters.] [1.7935e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 5e+1
# [00050.747 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# [00041.159 ms; 055 iters.] [1.0948e-6 err.] pagerankCuda [static]
# [00016.771 ms; 000 iters.] [3.4107e-6 err.] pagerankNvgraph [dynamic]
# [00003.100 ms; 005 iters.] [2.9557e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 1e+2
# [00050.811 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# [00041.132 ms; 055 iters.] [1.0711e-6 err.] pagerankCuda [static]
# [00021.139 ms; 000 iters.] [3.6037e-6 err.] pagerankNvgraph [dynamic]
# [00003.795 ms; 007 iters.] [3.6564e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 5e+2
# [00050.751 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# [00041.150 ms; 055 iters.] [1.1019e-6 err.] pagerankCuda [static]
# [00027.601 ms; 000 iters.] [3.7225e-6 err.] pagerankNvgraph [dynamic]
# [00006.194 ms; 011 iters.] [5.4797e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 1e+3
# [00050.783 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# [00041.275 ms; 055 iters.] [1.1030e-6 err.] pagerankCuda [static]
# [00030.140 ms; 000 iters.] [3.7582e-6 err.] pagerankNvgraph [dynamic]
# [00007.977 ms; 014 iters.] [6.2178e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 5e+3
# [00050.784 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# [00041.161 ms; 055 iters.] [1.1203e-6 err.] pagerankCuda [static]
# [00035.040 ms; 000 iters.] [3.8467e-6 err.] pagerankNvgraph [dynamic]
# [00013.958 ms; 022 iters.] [7.0924e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 1e+4
# [00050.767 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# [00041.185 ms; 055 iters.] [1.0687e-6 err.] pagerankCuda [static]
# [00036.815 ms; 000 iters.] [3.8981e-6 err.] pagerankNvgraph [dynamic]
# [00016.689 ms; 026 iters.] [7.2326e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 5e+4
# [00050.824 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# [00041.199 ms; 055 iters.] [1.1390e-6 err.] pagerankCuda [static]
# [00041.133 ms; 000 iters.] [4.0983e-6 err.] pagerankNvgraph [dynamic]
# [00022.985 ms; 034 iters.] [7.4402e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 1e+5
# [00050.931 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# [00041.229 ms; 055 iters.] [1.1307e-6 err.] pagerankCuda [static]
# [00042.930 ms; 000 iters.] [4.2112e-6 err.] pagerankNvgraph [dynamic]
# [00025.708 ms; 038 iters.] [7.5040e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 5e+5
# [00051.312 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# [00041.630 ms; 055 iters.] [1.1226e-6 err.] pagerankCuda [static]
# [00047.532 ms; 000 iters.] [4.6612e-6 err.] pagerankNvgraph [dynamic]
# [00032.234 ms; 046 iters.] [7.5906e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 1e+6
# [00051.579 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# [00042.021 ms; 055 iters.] [1.1029e-6 err.] pagerankCuda [static]
# [00049.934 ms; 000 iters.] [4.8865e-6 err.] pagerankNvgraph [dynamic]
# [00035.571 ms; 049 iters.] [7.6489e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 5e+6
# [00054.511 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# [00044.391 ms; 055 iters.] [1.1946e-6 err.] pagerankCuda [static]
# [00056.491 ms; 000 iters.] [5.3385e-6 err.] pagerankNvgraph [dynamic]
# [00043.534 ms; 056 iters.] [7.1349e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 1e+7
# [00059.160 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# [00048.480 ms; 056 iters.] [1.3263e-6 err.] pagerankCuda [static]
# [00062.007 ms; 000 iters.] [5.5029e-6 err.] pagerankNvgraph [dynamic]
# [00048.396 ms; 056 iters.] [6.9686e-6 err.] pagerankCuda [dynamic]
#
# # Batch size 5e+7
# [00076.627 ms; 000 iters.] [0.0000e+0 err.] pagerankNvgraph [static]
# [00064.327 ms; 058 iters.] [1.5363e-6 err.] pagerankCuda [static]
# [00077.659 ms; 000 iters.] [4.2718e-6 err.] pagerankNvgraph [dynamic]
# [00064.371 ms; 059 iters.] [4.4546e-6 err.] pagerankCuda [dynamic]
```

[![](https://i.imgur.com/kdiENBk.gif)][sheets]

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [Stanford Large Network Dataset Collection]

<br>
<br>

[![](https://i.imgur.com/68DVPzP.jpg)](https://www.youtube.com/watch?v=SoiKp2oSUl0)

[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[CSR]: https://github.com/puzzlef/pagerank-class-vs-csr
[scaled-fill]: https://github.com/puzzlef/pagerank-dynamic-adjust-ranks
[charts]: https://photos.app.goo.gl/7zTbHBXV6uh7FGyd8
[sheets]: https://docs.google.com/spreadsheets/d/1TPFX5al0-rlSde0xr7zlfCHNYEqxXSfS6P8QIa2dDsA/edit?usp=sharing
[Stanford Large Network Dataset Collection]: http://snap.stanford.edu/data/index.html
