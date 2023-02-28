Design of *CUDA-based* **Dynamic** *PageRank algorithm* for link analysis.

<br>


### Comparing with Static approach on Fixed graphs (Random batches)

This experiment ([compare-static-on-fixed]) was for comparing the performance
between, finding **static** pagerank of updated graph using
[nvGraph][pr-nvgraph], finding **incremental** pagerank of updated graph using
[nvGraph][pr-nvgraph], finding **static** pagerank of updated graph using
[CUDA], and finding **incremental** pagerank of updated graph using [CUDA]. Each
approach was attempted on a number of graphs, running each with multiple batch
sizes (`1`, `5`, `10`, `50`, ...). Each batch size was run with 5 different
updates to graph, and each specific update was run 5 times for each approach to
get a good time measure. On average, **incremental CUDA** pagerank is **faster**
than *static* approach.

[compare-static-on-fixed]: https://github.com/puzzlef/pagerank-cuda-dynamic/tree/compare-static-on-fixed

<br>


### Comparing with Static approach on Temporal graphs

This experiment ([compare-static-on-temporal]) was for comparing the performance
between, finding **static** pagerank of updated graph using
[nvGraph][pr-nvgraph], finding **incremental** pagerank of updated graph using
[nvGraph][pr-nvgraph], finding **static** pagerank of updated graph using
[CUDA], and finding **incremental** pagerank of updated graph using [CUDA]. Each
technique was attempted on different temporal graphs, updating each graph with
multiple batch sizes (`1`, `5`, `10`, `50`, ...). New edges are incrementally
added to the graph batch-by-batch until the entire graph is complete.
**Incremental** pagerank using **CUDA** is indeed **faster** than *static*
approach for many batch sizes. In order to measure error, [nvGraph] pagerank is
taken as a reference.

[![](https://i.imgur.com/rWozeTl.gif)][sheetp]
[![](https://i.imgur.com/G8p8oUV.gif)][sheetp]
[![](https://i.imgur.com/7lsmZmQ.gif)][sheetp]
[![](https://i.imgur.com/BdCuBvu.gif)][sheetp]
[![](https://i.imgur.com/CfuX4PQ.gif)][sheetp]
[![](https://i.imgur.com/2E0RgRL.gif)][sheetp]
[![](https://i.imgur.com/qPV9Uma.gif)][sheetp]
[![](https://i.imgur.com/aH9zxZd.gif)][sheetp]
[![](https://i.imgur.com/mldojLF.gif)][sheetp]
[![](https://i.imgur.com/ysjEXtr.gif)][sheetp]
[![](https://i.imgur.com/Ra94KyW.gif)][sheetp]
[![](https://i.imgur.com/La6xhcz.gif)][sheetp]
[![](https://i.imgur.com/ckynVDc.gif)][sheetp]
[![](https://i.imgur.com/FJ4SaaB.gif)][sheetp]

[![](https://i.imgur.com/wpyR9RZ.png)][sheetp]
[![](https://i.imgur.com/DPX5Bux.png)][sheetp]
[![](https://i.imgur.com/nFlvaCB.png)][sheetp]
[![](https://i.imgur.com/EsbvLtJ.png)][sheetp]
[![](https://i.imgur.com/ZEiXFMx.png)][sheetp]
[![](https://i.imgur.com/bJJFpd6.png)][sheetp]

[compare-static-on-temporal]: https://github.com/puzzlef/pagerank-cuda-dynamic/tree/compare-static-on-temporal

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [Managed memory vs cudaHostAlloc - TK1](https://forums.developer.nvidia.com/t/managed-memory-vs-cudahostalloc-tk1/34281)
- [Stanford Large Network Dataset Collection]

<br>
<br>


[![](https://i.imgur.com/98aAG4g.jpg)](https://www.youtube.com/watch?v=_iSPqH3tHLI)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
[![DOI](https://zenodo.org/badge/368720311.svg)](https://zenodo.org/badge/latestdoi/368720311)


[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://cstar.iiit.ac.in/~kkishore/
[Stanford Large Network Dataset Collection]: http://snap.stanford.edu/data/index.html
[nvGraph]: https://github.com/rapidsai/nvgraph
[pr-nvgraph]: https://github.com/puzzlef/pagerank-nvgraph
[CUDA]: https://github.com/puzzlef/pagerank-cuda
[charts]: https://photos.app.goo.gl/AFpUhz3EzohBmxQT7
[sheets]: https://docs.google.com/spreadsheets/d/1XnVgAdSDIVInJn9XeIxUdXRkmsPgwsjH_4rAjCUYwkc/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vT27-_m62PzGz27urZkXyqu-YrI050uY98J7eoxGDPoJw5Z73Kxhwa06TXwef6onhcHwkC9FrZdoMQD/pubhtml
