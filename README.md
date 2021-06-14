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
# Using graph sx-stackoverflow.txt ...
# Temporal edges: 63497051
# order: 2601977 size: 36233450 {}
#
# # Batch size 1e+0
# [00042.938 ms; 054 iters.] [0.0000e+0 err.] pagerankStatic
# [00000.935 ms; 001 iters.] [8.8715e-7 err.] pagerankDynamic
#
# # Batch size 5e+0
# [00042.942 ms; 054 iters.] [0.0000e+0 err.] pagerankStatic
# [00001.261 ms; 002 iters.] [1.3280e-6 err.] pagerankDynamic
#
# # Batch size 1e+1
# [00042.923 ms; 054 iters.] [0.0000e+0 err.] pagerankStatic
# [00001.664 ms; 003 iters.] [1.7694e-6 err.] pagerankDynamic
#
# # Batch size 5e+1
# [00042.931 ms; 055 iters.] [0.0000e+0 err.] pagerankStatic
# [00003.293 ms; 005 iters.] [3.0156e-6 err.] pagerankDynamic
#
# # Batch size 1e+2
# [00043.008 ms; 055 iters.] [0.0000e+0 err.] pagerankStatic
# [00004.040 ms; 007 iters.] [3.7266e-6 err.] pagerankDynamic
#
# # Batch size 5e+2
# [00042.919 ms; 055 iters.] [0.0000e+0 err.] pagerankStatic
# [00006.607 ms; 011 iters.] [5.6382e-6 err.] pagerankDynamic
#
# # Batch size 1e+3
# [00042.944 ms; 055 iters.] [0.0000e+0 err.] pagerankStatic
# [00008.470 ms; 014 iters.] [6.3836e-6 err.] pagerankDynamic
#
# # Batch size 5e+3
# [00042.783 ms; 055 iters.] [0.0000e+0 err.] pagerankStatic
# [00014.652 ms; 022 iters.] [7.1914e-6 err.] pagerankDynamic
#
# # Batch size 1e+4
# [00042.888 ms; 055 iters.] [0.0000e+0 err.] pagerankStatic
# [00017.527 ms; 026 iters.] [7.2976e-6 err.] pagerankDynamic
#
# # Batch size 5e+4
# [00042.856 ms; 055 iters.] [0.0000e+0 err.] pagerankStatic
# [00024.024 ms; 034 iters.] [7.5914e-6 err.] pagerankDynamic
#
# # Batch size 1e+5
# [00042.936 ms; 055 iters.] [0.0000e+0 err.] pagerankStatic
# [00026.837 ms; 038 iters.] [7.6032e-6 err.] pagerankDynamic
#
# # Batch size 5e+5
# [00043.262 ms; 055 iters.] [0.0000e+0 err.] pagerankStatic
# [00033.566 ms; 046 iters.] [7.7148e-6 err.] pagerankDynamic
#
# # Batch size 1e+6
# [00043.240 ms; 055 iters.] [0.0000e+0 err.] pagerankStatic
# [00036.810 ms; 049 iters.] [7.7823e-6 err.] pagerankDynamic
#
# # Batch size 5e+6
# [00045.606 ms; 055 iters.] [0.0000e+0 err.] pagerankStatic
# [00044.518 ms; 056 iters.] [6.9372e-6 err.] pagerankDynamic
#
# # Batch size 1e+7
# [00049.703 ms; 056 iters.] [0.0000e+0 err.] pagerankStatic
# [00049.136 ms; 056 iters.] [6.2674e-6 err.] pagerankDynamic
#
# # Batch size 5e+7
# [00064.434 ms; 058 iters.] [0.0000e+0 err.] pagerankStatic
# [00065.651 ms; 059 iters.] [3.2916e-6 err.] pagerankDynamic
```

[![](https://i.imgur.com/kdiENBk.gif)][sheets]
[![](https://i.imgur.com/f4ZrnBw.gif)][sheets]
[![](https://i.imgur.com/c5K8Gkc.gif)][sheets]
[![](https://i.imgur.com/5EzB4ig.gif)][sheets]
[![](https://i.imgur.com/Lb91hO8.gif)][sheets]
[![](https://i.imgur.com/JPvR8rW.gif)][sheets]
[![](https://i.imgur.com/9u02uwR.gif)][sheets]
[![](https://i.imgur.com/HXTRDe1.gif)][sheets]
[![](https://i.imgur.com/p7osdEa.gif)][sheets]
[![](https://i.imgur.com/dlQ2paE.gif)][sheets]
[![](https://i.imgur.com/ZpYASa4.gif)][sheets]
[![](https://i.imgur.com/1eq8VBL.gif)][sheets]
[![](https://i.imgur.com/5QqBoTg.gif)][sheets]
[![](https://i.imgur.com/Id3HGcn.gif)][sheets]

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
