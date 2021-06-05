Performance of static vs dynamic PageRank ([pull], [CSR], [scaled-fill]).

This experiment was for comparing the performance between:
1. Find static pagerank of updated graph.
2. Find dynamic pagerank, **scaling** old vertices, and using **1/N** for new vertices.

Both techniques were attempted on different temporal graphs, updating each
graph with multiple batch sizes. Batch sizes are always an order of 10. New
edges are incrementally added to the graph batch-by-batch until the entire
graph is complete. The speedup of dynamic pagerank tends to **~2x** of static
pagerank when batch size is **1000**. I was able to get [cool charts] for these
logs using [sheets], showing the comparision.

All outputs (including shortened versions) are saved in [out/](out/) and
outputs for `email-Eu-core-temporal` and `wiki-talk-temporal` are listed here.
The input data used for this experiment is available at the
[Stanford Large Network Dataset Collection].

<br>

```bash
$ g++ -O3 main.cxx
$ ./a.out ~/data/email-Eu-core-temporal.txt

# (SHORTENED)
# Using graph email-Eu-core-temporal.txt ...
# Temporal edges: 332335
# order: 986 size: 24929 {}
#
# # Batch size 1e+0
# [00000.498 ms; 027 iters.] [0.0000e+0 err.] pagerankStatic
# [00000.032 ms; 002 iters.] [6.4766e-7 err.] pagerankDynamic
#
# # Batch size 5e+0
# [00000.510 ms; 027 iters.] [0.0000e+0 err.] pagerankStatic
# [00000.065 ms; 003 iters.] [1.0864e-6 err.] pagerankDynamic
#
# # Batch size 1e+1
# [00000.501 ms; 027 iters.] [0.0000e+0 err.] pagerankStatic
# [00000.098 ms; 005 iters.] [1.5461e-6 err.] pagerankDynamic
#
# # Batch size 5e+1
# [00000.503 ms; 027 iters.] [0.0000e+0 err.] pagerankStatic
# [00000.218 ms; 011 iters.] [2.4047e-6 err.] pagerankDynamic
#
# # Batch size 1e+2
# [00000.502 ms; 027 iters.] [0.0000e+0 err.] pagerankStatic
# [00000.251 ms; 014 iters.] [2.4436e-6 err.] pagerankDynamic
#
# # Batch size 5e+2
# [00000.499 ms; 027 iters.] [0.0000e+0 err.] pagerankStatic
# [00000.308 ms; 016 iters.] [2.5226e-6 err.] pagerankDynamic
#
# # Batch size 1e+3
# [00000.501 ms; 027 iters.] [0.0000e+0 err.] pagerankStatic
# [00000.336 ms; 018 iters.] [2.4926e-6 err.] pagerankDynamic
#
# # Batch size 5e+3
# [00000.509 ms; 027 iters.] [0.0000e+0 err.] pagerankStatic
# [00000.391 ms; 020 iters.] [2.3684e-6 err.] pagerankDynamic
#
# # Batch size 1e+4
# [00000.518 ms; 027 iters.] [0.0000e+0 err.] pagerankStatic
# [00000.415 ms; 022 iters.] [2.1345e-6 err.] pagerankDynamic
#
# # Batch size 5e+4
# [00000.530 ms; 024 iters.] [0.0000e+0 err.] pagerankStatic
# [00000.481 ms; 022 iters.] [1.6988e-6 err.] pagerankDynamic
#
# # Batch size 1e+5
# [00000.577 ms; 024 iters.] [0.0000e+0 err.] pagerankStatic
# [00000.541 ms; 022 iters.] [1.5049e-6 err.] pagerankDynamic
```

[![](https://i.imgur.com/4tWWPOT.gif)][sheets]
[![](https://i.imgur.com/VAHYT9C.gif)][sheets]

<br>
<br>

```bash
$ g++ -O3 main.cxx
$ ./a.out ~/data/sx-stackoverflow.txt

# (SHORTENED)
# Using graph sx-stackoverflow.txt ...
# Temporal edges: 63497051
# order: 2601977 size: 36233450 {}
#
# # Batch size 1e+0
# [04657.920 ms; 056 iters.] [0.0000e+0 err.] pagerankStatic
# [00099.498 ms; 001 iters.] [9.6450e-7 err.] pagerankDynamic
#
# # Batch size 5e+0
# [04634.926 ms; 056 iters.] [0.0000e+0 err.] pagerankStatic
# [00135.350 ms; 002 iters.] [1.4472e-6 err.] pagerankDynamic
#
# # Batch size 1e+1
# [04648.379 ms; 056 iters.] [0.0000e+0 err.] pagerankStatic
# [00179.794 ms; 003 iters.] [1.9689e-6 err.] pagerankDynamic
#
# # Batch size 5e+1
# [04628.978 ms; 057 iters.] [0.0000e+0 err.] pagerankStatic
# [00341.603 ms; 006 iters.] [3.2815e-6 err.] pagerankDynamic
#
# # Batch size 1e+2
# [04609.713 ms; 057 iters.] [0.0000e+0 err.] pagerankStatic
# [00414.303 ms; 007 iters.] [4.0053e-6 err.] pagerankDynamic
#
# # Batch size 5e+2
# [04609.833 ms; 057 iters.] [0.0000e+0 err.] pagerankStatic
# [00685.551 ms; 011 iters.] [5.7854e-6 err.] pagerankDynamic
#
# # Batch size 1e+3
# [04628.448 ms; 057 iters.] [0.0000e+0 err.] pagerankStatic
# [00872.455 ms; 014 iters.] [7.5399e-6 err.] pagerankDynamic
#
# # Batch size 5e+3
# [04612.280 ms; 057 iters.] [0.0000e+0 err.] pagerankStatic
# [01532.175 ms; 022 iters.] [7.3422e-6 err.] pagerankDynamic
#
# # Batch size 1e+4
# [04636.627 ms; 057 iters.] [0.0000e+0 err.] pagerankStatic
# [01825.588 ms; 026 iters.] [7.2866e-6 err.] pagerankDynamic
#
# # Batch size 5e+4
# [04690.433 ms; 057 iters.] [0.0000e+0 err.] pagerankStatic
# [02556.521 ms; 034 iters.] [7.1117e-6 err.] pagerankDynamic
#
# # Batch size 1e+5
# [04637.379 ms; 057 iters.] [0.0000e+0 err.] pagerankStatic
# [02821.770 ms; 038 iters.] [1.1516e-5 err.] pagerankDynamic
#
# # Batch size 5e+5
# [04606.930 ms; 057 iters.] [0.0000e+0 err.] pagerankStatic
# [03491.683 ms; 046 iters.] [7.5847e-6 err.] pagerankDynamic
#
# # Batch size 1e+6
# [04590.247 ms; 056 iters.] [0.0000e+0 err.] pagerankStatic
# [03825.545 ms; 050 iters.] [7.3622e-6 err.] pagerankDynamic
#
# # Batch size 5e+6
# [04789.445 ms; 056 iters.] [0.0000e+0 err.] pagerankStatic
# [04697.050 ms; 056 iters.] [6.9308e-6 err.] pagerankDynamic
#
# # Batch size 1e+7
# [05223.199 ms; 057 iters.] [0.0000e+0 err.] pagerankStatic
# [05183.780 ms; 057 iters.] [6.0776e-6 err.] pagerankDynamic
#
# # Batch size 5e+7
# [06678.227 ms; 058 iters.] [0.0000e+0 err.] pagerankStatic
# [06742.952 ms; 059 iters.] [3.3479e-6 err.] pagerankDynamic
```

[![](https://i.imgur.com/3zo6nzy.gif)][sheets]
[![](https://i.imgur.com/v5y3qiY.gif)][sheets]

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [Stanford Large Network Dataset Collection]

<br>
<br>

[![](https://i.imgur.com/0TfMELc.jpg)](https://www.youtube.com/watch?v=npl0o3X7NTA)

[pull]: https://github.com/puzzlef/pagerank-push-vs-pull
[CSR]: https://github.com/puzzlef/pagerank-class-vs-csr
[scaled-fill]: https://github.com/puzzlef/pagerank-dynamic-adjust-ranks
[cool charts]: https://photos.app.goo.gl/dcQWY7z1HEdPAqre8
[sheets]: https://docs.google.com/spreadsheets/d/1b6fuE9dRbAbQanCl2rDXc-K2xpIUSg7Mw_dzVnFbkD8/edit?usp=sharing
["graphs"]: https://github.com/puzzlef/graphs
[Stanford Large Network Dataset Collection]: http://snap.stanford.edu/data/index.html
