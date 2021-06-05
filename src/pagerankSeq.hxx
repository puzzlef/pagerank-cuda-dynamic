#pragma once
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "pagerank.hxx"

using std::swap;




template <class T>
T pagerankTeleport(const vector<T>& r, const vector<int>& vfrom, const vector<int>& efrom, const vector<int>& vdata, int N, T p) {
  T a = (1-p)/N;
  for (int u=0; u<N; u++)
    if (vdata[u] == 0) a += p*r[u]/N;
  return a;
}

template <class T>
void pagerankFactor(vector<T>& a, const vector<int>& vfrom, const vector<int>& efrom, const vector<int>& vdata, int N, T p) {
  for (int u=0; u<N; u++) {
    int d = vdata[u];
    a[u] = d>0? p/d : 0;
  }
}

template <class T>
void pagerankCalculate(vector<T>& a, const vector<T>& c, const vector<int>& vfrom, const vector<int>& efrom, const vector<int>& vdata, int N, T c0) {
  for (int v=0; v<N; v++)
    a[v] = c0 + sumAt(c, slice(efrom, vfrom[v], vfrom[v+1]));
}

template <class T>
int pagerankSeqLoop(vector<T>& a, vector<T>& r, const vector<T>& f, vector<T>& c, const vector<int>& vfrom, const vector<int>& efrom, const vector<int>& vdata, int N, T p, T E, int L) {
  int l = 1;
  for (; l<L; l++) {
    T c0 = pagerankTeleport(r, vfrom, efrom, vdata, N, p);
    multiply(c, r, f);
    pagerankCalculate(a, c, vfrom, efrom, vdata, N, c0);
    T e1 = absError(a, r);
    if (e1 < E) break;
    swap(a, r);
  }
  return l;
}

template <class T>
int pagerankSeqCore(vector<T>& a, vector<T>& r, vector<T>& f, vector<T>& c, const vector<int>& vfrom, const vector<int>& efrom, const vector<int>& vdata, int N, const vector<T> *q, T p, T E, int L) {
  if (q) copy(r, *q);
  else fill(r, T(1)/N);
  pagerankFactor(f, vfrom, efrom, vdata, N, p);
  return pagerankSeqLoop(a, r, f, c, vfrom, efrom, vdata, N, p, E, L);
}


// Find pagerank using a single thread.
// @param xt transpose graph, with vertex-data=out-degree
// @param q initial ranks (optional)
// @param o options {damping=0.85, tolerance=1e-6, maxIterations=500}
// @returns {ranks, iterations, time}
template <class G, class T=float>
PagerankResult<T> pagerankSeq(const G& xt, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  T    p = o.damping;
  T    E = o.tolerance;
  int  L = o.maxIterations, l;
  auto vfrom = sourceOffsets(xt);
  auto efrom = destinationIndices(xt);
  auto vdata = vertexData(xt);
  int  N     = xt.order();
  vector<T> a(N), r(N), f(N), c(N);
  vector<T> *qc = q? new vector<T> : nullptr;
  if (q) *qc = compressContainer(xt, *q);
  float t = measureDuration([&]() { l = pagerankSeqCore(a, r, f, c, vfrom, efrom, vdata, N, qc, p, E, L); }, o.repeat);
  return {decompressContainer(xt, a), l, t};
}
