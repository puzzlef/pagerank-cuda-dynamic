#include <cmath>
#include <string>
#include <sstream>
#include <cstdio>
#include <iostream>
#include <utility>
#include <algorithm>
#include "src/main.hxx"

using namespace std;



template <class G, class T>
auto runPagerankCall(const char *name, const G& xt, const vector<T> *init, const vector<T> *ranks=nullptr) {
  int repeat = name? 5 : 1;
  auto a = pagerankSeq(xt, init, {repeat});
  auto e = absError(a.ranks, ranks? *ranks : a.ranks);
  if (name) { print(xt); printf(" [%09.3f ms; %03d iters.] [%.4e err.] %s\n", a.time, a.iterations, e, name); }
  return a;
}


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

    // Find static pagerank of updated graph.
    auto a1 = runPagerankCall("pagerankStatic", xt, initStatic);

    // Find dynamic pagerank, scaling old vertices, and using 1/N for new vertices.
    adjustRanks(ranksAdj, ranksOld, ksOld, ks, 0.0f, float(ksOld.size())/ks.size(), 1.0f/ks.size());
    auto a2 = runPagerankCall("pagerankDynamic", xt, initDynamic, &a1.ranks);

    // Skip some edges (to speed up execution)
    if (skip) {
      if (!readSnapTemporal(x, s, skip)) break;
      ks = vertices(x);
      xt = transposeWithDegree(x);
      a1 = runPagerankCall(nullptr, xt, initStatic);
    }

    ksOld = move(ks);
    ranksOld = move(a1.ranks);
  }
}


void runPagerank(const string& data, bool show) {
  int M = countLines(data), steps = 100;
  printf("Temporal edges: %d\n\n", M);
  for (int batch=1, i=0; batch<M; batch*=i&1? 2:5, i++) {
    int skip = max(M/steps - batch, 0);
    printf("# Batch size %.0e\n", (double) batch);
    runPagerankBatch(data, show, batch, skip);
    printf("\n");
  }
}


int main(int argc, char **argv) {
  char *file = argv[1];
  bool  show = argc > 2;
  printf("Using graph %s ...\n", file);
  string d = readFile(file);
  runPagerank(d, show);
  return 0;
}
