#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <cstdio>
#include <iostream>
#include <utility>
#include <algorithm>
#include "src/main.hxx"

using namespace std;




#define TYPE float


void runPagerankBatch(const string& data, int repeat, int batch, int skip) {
  vector<TYPE> r1;
  vector<TYPE> *init = nullptr;
  PagerankOptions<TYPE> o = {repeat};

  DiGraph<> x;
  stringstream s(data);
  while (true) {
    // Skip some edges (to speed up execution)
    if (skip>0 && !readSnapTemporal(x, s, skip)) break;
    auto ksOld = vertices(x);
    auto xt = transposeWithDegree(x);
    auto a0 = pagerankCuda(xt, init, o);
    auto r0 = move(a0.ranks);

    // Read edges for this batch.
    if (!readSnapTemporal(x, s, batch)) break;
    auto ks = vertices(x);
    xt = transposeWithDegree(x);
    r1.resize(x.span());

    // Adjust ranks.
    adjustRanks(r1, r0, ksOld, ks, 0.0f, float(ksOld.size())/ks.size(), 1.0f/ks.size());

    // Find static pagerank of updated graph using nvGraph.
    auto a1 = pagerankNvgraph(xt, init, o);
    auto e1 = l1Norm(a1.ranks, a1.ranks);
    print(xt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankNvgraph [static]\n", a1.time, a1.iterations, e1);

    // Find static pagerank of updated graph using CUDA.
    auto a2 = pagerankCuda(xt, init, o);
    auto e2 = l1Norm(a2.ranks, a1.ranks);
    print(xt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankCuda [static]\n", a2.time, a2.iterations, e2);

    // Find incremental pagerank of updated graph using nvGraph.
    auto a3 = pagerankNvgraph(xt, &r1, o);
    auto e3 = l1Norm(a3.ranks, a1.ranks);
    print(xt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankNvgraph [incremental]\n", a3.time, a3.iterations, e3);

    // Find incremental pagerank of updated graph using CUDA.
    auto a4 = pagerankCuda(xt, &r1, o);
    auto e4 = l1Norm(a4.ranks, a1.ranks);
    print(xt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] pagerankCuda [incremental]\n", a4.time, a4.iterations, e4);
  }
}


void runPagerank(const string& data, int repeat) {
  int M = countLines(data), steps = 100;
  printf("Temporal edges: %d\n", M);
  for (int batch=10, i=0; batch<M; batch*=i&1? 2:5, i++) {
    int skip = max(M/steps - batch, 0);
    printf("\n# Batch size %.0e\n", (double) batch);
    runPagerankBatch(data, repeat, skip, batch);
  }
}


int main(int argc, char **argv) {
  char *file = argv[1];
  int repeat = argc>2? stoi(argv[2]) : 5;
  printf("Using graph %s ...\n", file);
  string d = readFile(file);
  runPagerank(d, repeat);
  printf("\n");
  return 0;
}
