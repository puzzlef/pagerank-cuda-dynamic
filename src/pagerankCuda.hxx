#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "pagerank.hxx"

using std::vector;
using std::swap;
using std::min;




// PAGERANK HELPERS
// ----------------

template <class G>
int pagerankSwitchPoint(const G& xt, const vector<int>& ks) {
  int lim = BLOCK_DIM_T, N = ks.size();
  int deg = int(0.5 * BLOCK_DIM_B);
  int a = lower_bound(ks.begin(), ks.end(), deg, [&](int u, int d) {
    return xt.degree(u) < d;
  }) - ks.begin();
  return a<lim? 0 : (N-a<lim? N : a);
}




// PAGERANK KERNELS
// ----------------

template <class T>
__global__ void pagerankFactorKernel(T *a, int *vdata, int v, int V, T p) {
  DEFINE(t, b, B, G);
  for (v+=B*b+t; v<V; v+=G*B) {
    int d = vdata[v];
    a[v] = d>0? p/d : 0;
  }
}


template <class T>
__global__ void pagerankBlockKernel(T *a, T *r, T *c, int *vfrom, int *efrom, int v, int V, T c0) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM_B];

  for (v+=b; v<V; v+=G) {
    int ebgn = vfrom[v];
    int ideg = vfrom[v+1]-vfrom[v];
    cache[t] = sumAtKernelLoop(c, efrom+ebgn, ideg, t, B);
    sumKernelReduce(cache, B, t);
    if (t == 0) a[v] = c0 + cache[0];
  }
}


template <class T>
__global__ void pagerankThreadKernel(T *a, T *r, T *c, int *vfrom, int *efrom, int v, int V, T c0) {
  DEFINE(t, b, B, G);

  for (v+=B*b+t; v<V; v+=G*B) {
    int ebgn = vfrom[v];
    int ideg = vfrom[v+1]-vfrom[v];
    a[v] = c0 + sumAtKernelLoop(c, efrom+ebgn, ideg, 0, 1);
  }
}


template<class T>
void pagerankBlockKernelCall(T *a, T *r, T *c, int *vfrom, int *efrom, int v, int V, T c0) {
  int B = BLOCK_DIM_B;
  int G = min(ceilDiv(V-v, B), GRID_DIM_B);
  pagerankBlockKernel<<<G, B>>>(a, r, c, vfrom, efrom, v, V, c0);
}

template<class T>
void pagerankThreadKernelCall(T *a, T *r, T *c, int *vfrom, int *efrom, int v, int V, T c0) {
  int B = BLOCK_DIM_T;
  int G = min(ceilDiv(V-v, B), GRID_DIM_T);
  pagerankThreadKernel<<<G, B>>>(a, r, c, vfrom, efrom, v, V, c0);
}

template <class T, class J>
void pagerankKernelWave(T *a, T *r, T *c, int *vfrom, int *efrom, int v, J&& ns, T c0) {
  for (int n : ns) {
    if (n>0)      pagerankBlockKernelCall (a, r, c, vfrom, efrom, v, v+n, c0);
    else if (n<0) pagerankThreadKernelCall(a, r, c, vfrom, efrom, v, v-n, c0);
    v += abs(n);
  }
}




// PAGERANK CUDA
// -------------

template <class T, class J>
int pagerankCudaLoop(T *e, T *r0, T *eD, T *r0D, T *&aD, T *&rD, T *cD, T *fD, int *vfromD, int *efromD, int *vdataD, int v, J&& ns, int N, T p, T E, int L) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  int G1 = G * sizeof(T), l = 1;
  for (; l<L; l++) {
    sumIfNotKernel<<<G, B>>>(r0D, rD, vdataD, N);
    multiplyKernel<<<G, B>>>(cD,  rD, fD,     N);
    TRY( cudaMemcpy(r0, r0D, G1, cudaMemcpyDeviceToHost) );
    T c0 = (1-p)/N + p*sum(r0, G)/N;
    pagerankKernelWave(aD, rD, cD, vfromD, efromD, v, ns, c0);
    l1NormKernel<<<G, B>>>(eD, rD, aD, N);
    TRY( cudaMemcpy(e, eD, G1, cudaMemcpyDeviceToHost) );
    T e1 = sum(e, G);
    if (e1 < E) break;
    swap(aD, rD);
  }
  return l;
}


template <class T, class J>
int pagerankCudaCore(T *e, T *r0, T *eD, T *r0D, T *&aD, T *&rD, T *cD, T *fD, int *vfromD, int *efromD, int *vdataD, J&& ns, int N, T p, T E, int L) {
  int B = BLOCK_DIM;
  int G = min(ceilDiv(N, B), GRID_DIM);
  pagerankFactorKernel<<<G, B>>>(fD, vdataD, 0, N, p);
  return pagerankCudaLoop(e, r0, eD, r0D, aD, rD, cD, fD, vfromD, efromD, vdataD, 0, ns, N, p, E, L);
}


template <class H, class T=float>
PagerankResult<T> pagerankCuda(const H& xt, const vector<T> *q=nullptr, PagerankOptions<T> o=PagerankOptions<T>()) {
  T    p  = o.damping;
  T    E  = o.tolerance;
  int  L  = o.maxIterations, l;
  int  N     = xt.order();
  auto ks    = verticesByDegree(xt);
  auto vfrom = sourceOffsets(xt, ks);
  auto efrom = destinationIndices(xt, ks);
  auto vdata = vertexData(xt, ks);
  int  S     = pagerankSwitchPoint(xt, ks);
  int VFROM1 = vfrom.size() * sizeof(int);
  int EFROM1 = efrom.size() * sizeof(int);
  int VDATA1 = vdata.size() * sizeof(int);
  int G1     = GRID_DIM * sizeof(T);
  int N1     = N        * sizeof(T);
  vector<int> ns {-S, N-S};
  vector<T> a(N), r(N);

  T *e,  *r0;
  T *eD, *r0D, *fD, *rD, *cD, *aD;
  int *vfromD, *efromD, *vdataD;
  // TRY( cudaProfilerStart() );
  TRY( cudaSetDeviceFlags(cudaDeviceMapHost) );
  TRY( cudaHostAlloc(&e,  G1, cudaHostAllocDefault) );
  TRY( cudaHostAlloc(&r0, G1, cudaHostAllocDefault) );
  TRY( cudaMalloc(&eD,  G1) );
  TRY( cudaMalloc(&r0D, G1) );
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
    mark([&] { l = pagerankCudaCore(e, r0, eD, r0D, aD, rD, cD, fD, vfromD, efromD, vdataD, ns, N, p, E, L); });
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
