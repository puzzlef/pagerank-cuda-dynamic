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

# ...
```

[![](https://i.imgur.com/4tWWPOT.gif)][sheets]
[![](https://i.imgur.com/VAHYT9C.gif)][sheets]

<br>
<br>

```bash
$ g++ -O3 main.cxx
$ ./a.out ~/data/sx-stackoverflow.txt

# ...
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
