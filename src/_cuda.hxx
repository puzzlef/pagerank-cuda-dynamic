#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "_main.hxx"

using std::min;
using std::fprintf;
using std::exit;




// LAUNCH CONFIG
// -------------

// Limits
#define BLOCK_LIMIT 1024
#define GRID_LIMIT  65535

// For map-like operations
#define BLOCK_DIM_M 256
#define GRID_DIM_M  GRID_LIMIT

// For reduce-like operations (memcpy)
#define BLOCK_DIM_RM 128
#define GRID_DIM_RM  1024

// For reduce-like operations (in-place)
#define BLOCK_DIM_RI 128
#define GRID_DIM_RI  1024

// Maximum for map-like operations
#define BLOCK_MAX_M 256
#define GRID_MAX_M  GRID_LIMIT

// Maximum for reduce-like operations
#define BLOCK_MAX_R 128
#define GRID_MAX_R  1024

// Preffered maximum
#define BLOCK_MAX 512
#define GRID_MAX  GRID_LIMIT




// TRY
// ---
// Log error if CUDA function call fails.

#ifndef TRY_CUDA
void tryCuda(cudaError err, const char* exp, const char* func, int line, const char* file) {
  if (err == cudaSuccess) return;
  fprintf(stderr,
    "%s: %s\n"
    "  in expression %s\n"
    "  at %s:%d in %s\n",
    cudaGetErrorName(err), cudaGetErrorString(err), exp, func, line, file);
  exit(err);
}

#define TRY_CUDA(exp) tryCuda(exp, #exp, __func__, __LINE__, __FILE__)
#endif

#ifndef TRY
#define TRY(exp) TRY_CUDA(exp)
#endif




// DEFINE
// ------
// Define thread, block variables.

#ifndef DEFINE_CUDA
#define DEFINE_CUDA(t, b, B, G) \
  const int t = threadIdx.x; \
  const int b = blockIdx.x; \
  const int B = blockDim.x; \
  const int G = gridDim.x;
#define DEFINE_CUDA2D(tx, ty, bx, by, BX, BY, GX, GY) \
  const int tx = threadIdx.x; \
  const int ty = threadIdx.y; \
  const int bx = blockIdx.x; \
  const int by = blockIdx.y; \
  const int BX = blockDim.x; \
  const int BY = blockDim.y; \
  const int GX = gridDim.x;  \
  const int GY = gridDim.y;
#endif

#ifndef DEFINE
#define DEFINE(t, b, B, G) \
  DEFINE_CUDA(t, b, B, G)
#define DEFINE2D(tx, ty, bx, by, BX, BY, GX, GY) \
  DEFINE_CUDA2D(tx, ty, bx, by, BX, BY, GX, GY)
#endif




// UNUSED
// ------
// Mark CUDA kernel variables as unused.

template <class T>
__device__ void unusedCuda(T&&) {}

#ifndef UNUSED_CUDA
#define UNUSED_CUDA(...) ARG_CALL(unusedCuda, ##__VA_ARGS__)
#endif

#ifndef UNUSED
#define UNUSED UNUSED_CUDA
#endif




// REMOVE IDE SQUIGGLES
// --------------------

#ifndef __SYNCTHREADS
void __syncthreads();
#define __SYNCTHREADS() __syncthreads()
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __shared__
#define __shared__
#endif




// REDUCE
// ------

int reduceSizeCu(int N) {
  int B = BLOCK_DIM_RM;
  int G = min(ceilDiv(N, B), GRID_DIM_RM);
  return G;
}




// FILL
// ----

template <class T>
__device__ void fillKernelLoop(T *a, int N, T v, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] = v;
}


template <class T>
__global__ void fillKernel(T *a, int N, T v) {
  DEFINE(t, b, B, G);
  fillKernelLoop(a, N, v, B*b+t, G*B);
}


template <class T>
void fillCu(T *a, int N, T v) {
  int B = BLOCK_DIM_M;
  int G = min(ceilDiv(N, B), GRID_DIM_M);
  fillKernel<<<G, B>>>(a, N, v);
}




// FILL-AT
// -------

template <class T>
__device__ void fillAtKernelLoop(T *a, T v, const int *is, int IS, int i, int DI) {
  for (; i<IS; i+=DI)
    a[is[i]] = v;
}


template <class T>
__global__ void fillAtKernel(T *a, T v, const int *is, int IS) {
  DEFINE(t, b, B, G);
  fillAtKernelLoop(a, v, is, IS, B*b+t, G*B);
}


template <class T>
void fillAtCu(T *a, T v, const int *is, int IS) {
  int B = BLOCK_DIM_M;
  int G = min(ceilDiv(IS, B), GRID_DIM_M);
  fillAtKernel<<<G, B>>>(a, v, is, IS);
}




// SUM
// ---

template <class T>
__device__ void sumKernelReduce(T* a, int N, int i) {
  __syncthreads();
  for (N=N/2; N>0; N/=2) {
    if (i < N) a[i] += a[N+i];
    __syncthreads();
  }
}


template <class T>
__device__ T sumKernelLoop(const T *x, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    a += x[i];
  return a;
}


template <class T>
__global__ void sumKernel(T *a, const T *x, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_MAX_R];
  cache[t] = sumKernelLoop(x, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}


template <class T>
void sumMemcpyCu(T *a, const T *x, int N) {
  int B = BLOCK_DIM_RM;
  int G = min(ceilDiv(N, B), GRID_DIM_RM);
  sumKernel<<<G, B>>>(a, x, N);
}

template <class T>
void sumInplaceCu(T *a, const T *x, int N) {
  int B = BLOCK_DIM_RI;
  int G = min(ceilDiv(N, B), GRID_DIM_RI);
  sumKernel<<<G, B>>>(a, x, N);
  sumKernel<<<1, G>>>(a, a, G);
}

template <class T>
void sumCu(T *a, const T *x, int N) {
  sumMemcpyCu(a, x, N);
}




// SUM-AT
// ------

template <class T>
__device__ T sumAtKernelLoop(const T *x, const int *is, int IS, int i, int DI) {
  T a = T();
  for (; i<IS; i+=DI)
    a += x[is[i]];
  return a;
}


template <class T>
__global__ void sumAtKernel(T *a, const T *x, const T *is, int IS) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_MAX_R];
  cache[t] = sumAtKernelLoop(x, is, IS, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}


template <class T>
void sumAtMemcpyCu(T *a, const T *x, const T *is, int IS) {
  int B = BLOCK_DIM_RM;
  int G = min(ceilDiv(IS, B), GRID_DIM_RM);
  sumAtKernel<<<G, B>>>(a, x, is, IS);
}

template <class T>
void sumAtInplaceCu(T *a, const T *x, const T *is, int IS) {
  int B = BLOCK_DIM_RI;
  int G = min(ceilDiv(IS, B), GRID_DIM_RI);
  sumAtKernel<<<G, B>>>(a, x, is, IS);
  sumKernel<<<1, G>>>(a, a, G);
}

template <class T>
void sumAtCu(T *a, const T *x, const T *is, int IS) {
  sumAtMemcpyCu(a, x, is, IS);
}




// SUM-IF-NOT
// ----------

template <class T, class C>
__device__ T sumIfNotKernelLoop(const T *x, const C *cs, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    if (!cs[i]) a += x[i];
  return a;
}


template <class T, class C>
__global__ void sumIfNotKernel(T *a, const T *x, const C *cs, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_MAX_R];
  cache[t] = sumIfNotKernelLoop(x, cs, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}


template <class T, class C>
void sumIfNotMemcpyCu(T *a, const T *x, const C *cs, int N) {
  int B = BLOCK_DIM_RM;
  int G = min(ceilDiv(N, B), GRID_DIM_RM);
  sumIfNotKernel<<<G, B>>>(a, x, cs, N);
}

template <class T, class C>
void sumIfNotInplaceCu(T *a, const T *x, const C *cs, int N) {
  int B = BLOCK_DIM_RI;
  int G = min(ceilDiv(N, B), GRID_DIM_RI);
  sumIfNotKernel<<<G, B>>>(a, x, cs, N);
  sumKernel<<<1, G>>>(a, a, G);
}

template <class T, class C>
void sumIfNotCu(T *a, const T *x, const C *cs, int N) {
  sumIfNotMemcpyCu(a, x, cs, N);
}




// L1-NORM
// -------

template <class T>
__device__ T l1NormKernelLoop(const T *x, const T *y, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    a += abs(x[i] - y[i]);
  return a;
}


template <class T>
__global__ void l1NormKernel(T *a, const T *x, const T *y, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_MAX_R];
  cache[t] = l1NormKernelLoop(x, y, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}


template <class T>
void l1NormMemcpyCu(T *a, const T *x, const T *y, int N) {
  int B = BLOCK_DIM_RM;
  int G = min(ceilDiv(N, B), GRID_DIM_RM);
  l1NormKernel<<<G, B>>>(a, x, y, N);
}

template <class T>
void l1NormInplaceCu(T *a, const T *x, const T *y, int N) {
  int B = BLOCK_DIM_RI;
  int G = min(ceilDiv(N, B), GRID_DIM_RI);
  l1NormKernel<<<G, B>>>(a, x, y, N);
  sumKernel<<<1, G>>>(a, a, G);
}

template <class T>
void l1NormCu(T *a, const T *x, const T *y, int N) {
  l1NormMemcpyCu(a, x, y, N);
}




// MULTIPLY
// --------

template <class T>
__device__ void multiplyKernelLoop(T *a, const T *x, const T *y, int N, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] = x[i] * y[i];
}


template <class T>
__global__ void multiplyKernel(T *a, const T *x, const T* y, int N) {
  DEFINE(t, b, B, G);
  multiplyKernelLoop(a, x, y, N, B*b+t, G*B);
}


template <class T>
void multiplyCu(T *a, const T *x, const T* y, int N) {
  int B = BLOCK_DIM_M;
  int G = min(ceilDiv(N, B), GRID_DIM_M);
  multiplyKernel<<<G, B>>>(a, x, y, N);
}
