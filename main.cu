#include <cmath>
#include <string>
#include <sstream>
#include <cstdio>
#include <iostream>
#include <utility>
#include <algorithm>
#include "src/main.hxx"

using namespace std;




#define REPEAT 5

void runPagerankBatch(const string& data, bool show, int batch, int skip) {
  vector<float>  ranksOld, ranksAdj;
  vector<float> *initStatic  = nullptr;
  vector<float> *initDynamic = &ranksAdj;

  DiGraph<> x;
  stringstream s(data);
  auto ksOld = vertices(x);
  while(readSnapTemporal(x, s, batch)) {
    auto ks = vertices(x);
    auto xt = transposeWithDegree(x);
    ranksAdj.resize(x.span());

    // Find static pagerank of updated graph using nvGraph.
    auto a1 = pagerankNvgraph(xt, initStatic, {REPEAT});
    auto e1 = l1Norm(a1.ranks, a1.ranks);
    print(xt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankNvgraph [static]\n", a1.time, a1.iterations, e1);

    // Find static pagerank of updated graph using CUDA.
    auto a2 = pagerankCuda(xt, initStatic, {REPEAT});
    auto e2 = l1Norm(a2.ranks, a1.ranks);
    print(xt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankCuda [static]\n", a2.time, a2.iterations, e2);

    // Find dynamic pagerank of updated graph using nvGraph.
    adjustRanks(ranksAdj, ranksOld, ksOld, ks, 0.0f, float(ksOld.size())/ks.size(), 1.0f/ks.size());
    auto a3 = pagerankNvgraph(xt, initDynamic, {REPEAT});
    auto e3 = l1Norm(a3.ranks, a1.ranks);
    print(xt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankNvgraph [dynamic]\n", a3.time, a3.iterations, e3);

    // Find dynamic pagerank of updated graph using CUDA.
    auto a4 = pagerankCuda(xt, initDynamic, {REPEAT});
    auto e4 = l1Norm(a4.ranks, a1.ranks);
    print(xt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankCuda [dynamic]\n", a4.time, a4.iterations, e4);

    // Skip some edges (to speed up execution)
    if (skip) {
      if (!readSnapTemporal(x, s, skip)) break;
      ks = vertices(x);
      xt = transposeWithDegree(x);
      a1 = pagerankCuda(xt, initStatic);
    }

    ksOld = move(ks);
    ranksOld = move(a1.ranks);
  }
}


void runPagerank(const string& data, bool show) {
  int M = countLines(data), steps = 100;
  printf("Temporal edges: %d\n", M);
  for (int batch=1, i=0; batch<M; batch*=i&1? 2:5, i++) {
    int skip = max(M/steps - batch, 0);
    printf("\n# Batch size %.0e\n", (double) batch);
    runPagerankBatch(data, show, batch, skip);
  }
}


int main(int argc, char **argv) {
  char *file = argv[1];
  bool  show = argc > 2;
  printf("Using graph %s ...\n", file);
  string d = readFile(file);
  runPagerank(d, show);
  printf("\n");
  return 0;
}
