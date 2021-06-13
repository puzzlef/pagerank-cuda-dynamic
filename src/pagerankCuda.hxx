#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "pagerank.hxx"

using std::vector;
using std::partition;
using std::swap;
using std::min;




// PAGERANK-FACTOR
// ---------------

template <class T>
__global__ void pagerankFactorKernel(T *a, const int *vdata, int i, int n, T p) {
  DEFINE(t, b, B, G);
  for (int v=i+B*b+t; v<i+n; v+=G*B) {
    int d = vdata[v];
    a[v] = d>0? p/d : 0;
  }
}

template <class T>
void pagerankFactorCu(T *a, const int *vdata, int i, int n, T p) {
  int B = BLOCK_DIM_M;
  int G = min(ceilDiv(n, B), GRID_DIM_M);
  pagerankFactorKernel<<<G, B>>>(a, vdata, i, n, p);
}




// PAGERANK-BLOCK
// --------------

template <class T>
__global__ void pagerankBlockKernel(T *a, const T *r, const T *c, const int *vfrom, const int *efrom, int i, int n, T c0) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_LIMIT];
  for (int v=i+b; v<i+n; v+=G) {
    int ebgn = vfrom[v];
    int ideg = vfrom[v+1]-vfrom[v];
    cache[t] = sumAtKernelLoop(c, efrom+ebgn, ideg, t, B);
    sumKernelReduce(cache, B, t);
    if (t == 0) a[v] = c0 + cache[0];
  }
}

template <class T>
void pagerankBlockCu(T *a, const T *r, const T *c, const int *vfrom, const int *efrom, int i, int n, T c0) {
  int B = BLOCK_DIM_PRB;
  int G = min(n, GRID_DIM_PRB);
  pagerankBlockKernel<<<G, B>>>(a, r, c, vfrom, efrom, i, n, c0);
}




// PAGERANK-THREAD
// ---------------

template <class T>
__global__ void pagerankThreadKernel(T *a, const T *r, const T *c, const int *vfrom, const int *efrom, int i, int n, T c0) {
  DEFINE(t, b, B, G);
  for (int v=i+B*b+t; v<i+n; v+=G*B) {
    int ebgn = vfrom[v];
    int ideg = vfrom[v+1]-vfrom[v];
    a[v] = c0 + sumAtKernelLoop(c, efrom+ebgn, ideg, 0, 1);
  }
}

template <class T>
void pagerankThreadCu(T *a, const T *r, const T *c, const int *vfrom, const int *efrom, int i, int n, T c0) {
  int B = BLOCK_DIM_PRT;
  int G = min(ceilDiv(n, B), GRID_DIM_PRT);
  pagerankThreadKernel<<<G, B>>>(a, r, c, vfrom, efrom, i, n, c0);
}




// PAGERANK-SWITCHED
// -----------------


template <class T>
void pagerankSwitchedBlockCu(T *a, const T *r, const T *c, const int *vfrom, const int *efrom, int i, int n, T c0) {
  int B = BLOCK_DIM_PRSB;
  int G = min(n, GRID_DIM_PRSB);
  pagerankBlockKernel<<<G, B>>>(a, r, c, vfrom, efrom, i, n, c0);
}

template <class T>
void pagerankSwitchedThreadCu(T *a, const T *r, const T *c, const int *vfrom, const int *efrom, int i, int n, T c0) {
  int B = BLOCK_DIM_PRST;
  int G = min(ceilDiv(n, B), GRID_DIM_PRST);
  pagerankThreadKernel<<<G, B>>>(a, r, c, vfrom, efrom, i, n, c0);
}

template <class T, class J>
void pagerankSwitchedCu(T *a, const T *r, const T *c, const int *vfrom, const int *efrom, int i, J&& ns, T c0) {
  for (int n : ns) {
    if (n>0) pagerankSwitchedBlockCu (a, r, c, vfrom, efrom, i,  n, c0);
    else     pagerankSwitchedThreadCu(a, r, c, vfrom, efrom, i, -n, c0);
    i += abs(n);
  }
}

template <class G, class T=float>
int pagerankSwitchPoint(const G& xt) {
  int a = countIf(xt.vertices(), [&](int u) { return xt.degree(u) < PAGERANK_SWITCH_DEGREE; });
  int L = PAGERANK_SWITCH_LIMIT, N = xt.order();
  return a<L? 0 : (N-a<L? N : a);
}

void pagerankAddStep(vector<int>& a, int n) {
  if (a.empty() || sgn(a.back()) != sgn(n)) a.push_back(n);
  else a.back() += n;
}

template <class G, class T=float>
auto pagerankWave(const G& xt) {
  vector<int> a;
  int N = xt.order();
  int s = pagerankSwitchPoint(xt);
  if (s)   pagerankAddStep(a,  -s);
  if (N-s) pagerankAddStep(a, N-s);
  return a;
}




// PAGERANK (CUDA)
// ---------------

template <class T, class J>
int pagerankCudaLoop(T *e, T *r0, T *eD, T *r0D, T *&aD, T *&rD, T *cD, const T *fD, const int *vfromD, const int *efromD, const int *vdataD, int i, J&& ns, int N, T p, T E, int L) {
  int n = sumAbs(ns);
  int R = reduceSizeCu(n);
  size_t R1 = R * sizeof(T);
  int l = 1;
  for (; l<L; l++) {
    sumIfNotCu(r0D, rD, vdataD, N);
    multiplyCu(cD, rD, fD, n);
    TRY( cudaMemcpy(r0, r0D, R1, cudaMemcpyDeviceToHost) );
    T c0 = (1-p)/N + p*sum(r0, R)/N;
    pagerankSwitchedCu(aD, rD, cD, vfromD, efromD, i, ns, c0);
    l1NormCu(eD, rD, aD, n);
    TRY( cudaMemcpy(e, eD, R1, cudaMemcpyDeviceToHost) );
    T e1 = sum(e, R);
    if (e1 < E) break;
    swap(aD, rD);
  }
  return l;
}


template <class H, class T=float>
PagerankResult<T> pagerankCuda(H& xt, const vector<T> *q=nullptr, PagerankOptions<T> o=PagerankOptions<T>()) {
  T    p   = o.damping;
  T    E   = o.tolerance;
  int  L   = o.maxIterations, l;
  int  N   = xt.order();
  int  R   = reduceSizeCu(N);
  auto fm  = [](int u) { return u; };
  auto fp  = [&](auto ib, auto ie) {
    partition(ib, ie, [&](int u) { return xt.degree(u) < PAGERANK_SWITCH_DEGREE; });
  };
  auto ns    = pagerankWave(xt);
  auto ks    = vertices(xt, fm, fp);
  auto vfrom = sourceOffsets(xt, ks);
  auto efrom = destinationIndices(xt, ks);
  auto vdata = vertexData(xt, ks);
  int VFROM1 = vfrom.size() * sizeof(int);
  int EFROM1 = efrom.size() * sizeof(int);
  int VDATA1 = vdata.size() * sizeof(int);
  int N1 = N * sizeof(T);
  int R1 = R * sizeof(T);
  vector<T> a(N), r(N);

  T *e,  *r0;
  T *eD, *r0D, *fD, *rD, *cD, *aD;
  int *vfromD, *efromD, *vdataD;
  // TRY( cudaProfilerStart() );
  TRY( cudaSetDeviceFlags(cudaDeviceMapHost) );
  TRY( cudaHostAlloc(&e,  R1, cudaHostAllocDefault) );
  TRY( cudaHostAlloc(&r0, R1, cudaHostAllocDefault) );
  TRY( cudaMalloc(&eD,  R1) );
  TRY( cudaMalloc(&r0D, R1) );
  TRY( cudaMalloc(&aD, N1) );
  TRY( cudaMalloc(&rD, N1) );
  TRY( cudaMalloc(&cD, N1) );
  TRY( cudaMalloc(&fD, N1) );
  TRY( cudaMalloc(&vfromD, VFROM1) );
  TRY( cudaMalloc(&efromD, EFROM1) );
  TRY( cudaMalloc(&vdataD, VDATA1) );
  TRY( cudaMemcpy(vfromD, vfrom.data(), VFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(efromD, efrom.data(), EFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(vdataD, vdata.data(), VDATA1, cudaMemcpyHostToDevice) );

  float t = measureDurationMarked([&](auto mark) {
    if (q) r = compressContainer(xt, *q, ks);
    else fill(r, T(1)/N);
    TRY( cudaMemcpy(rD, r.data(), N1, cudaMemcpyHostToDevice) );
    mark([&] { pagerankFactorCu(fD, vdataD, 0, N, p); });
    mark([&] { l = pagerankCudaLoop(e, r0, eD, r0D, aD, rD, cD, fD, vfromD, efromD, vdataD, 0, ns, N, p, E, L); });
  }, o.repeat);
  TRY( cudaMemcpy(a.data(), aD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFreeHost(e) );
  TRY( cudaFreeHost(r0) );
  TRY( cudaFree(eD) );
  TRY( cudaFree(r0D) );
  TRY( cudaFree(aD) );
  TRY( cudaFree(rD) );
  TRY( cudaFree(cD) );
  TRY( cudaFree(fD) );
  TRY( cudaFree(vfromD) );
  TRY( cudaFree(efromD) );
  TRY( cudaFree(vdataD) );
  // TRY( cudaProfilerStop() );
  return {decompressContainer(xt, a, ks), l, t};
}
