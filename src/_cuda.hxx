#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

using std::fprintf;
using std::exit;




// LAUNCH CONFIG
// -------------

// For regular data
#define BLOCK_DIM 256
#define GRID_DIM  4096




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
  int t = threadIdx.x; \
  int b = blockIdx.x; \
  int B = blockDim.x; \
  int G = gridDim.x;
#define DEFINE_CUDA2D(tx, ty, bx, by, BX, BY, GX, GY) \
  int tx = threadIdx.x; \
  int ty = threadIdx.y; \
  int bx = blockIdx.x; \
  int by = blockIdx.y; \
  int BX = blockDim.x; \
  int BY = blockDim.y; \
  int GX = gridDim.x;  \
  int GY = gridDim.y;
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




// FILL (CUDA)
// -----------

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




// FILL-AT (CUDA)
// --------------

template <class T>
__device__ void fillAtKernelLoop(T *a, T v, int *is, int IS, int i, int DI) {
  for (; i<IS; i+=DI)
    a[is[i]] = v;
}


template <class T>
__global__ void fillAtKernel(T *a, T v, int *is, int IS) {
  DEFINE(t, b, B, G);

  fillAtKernelLoop(a, v, is, IS, B*b+t, G*B);
}




// SUM (CUDA)
// ----------

template <class T>
__device__ void sumKernelReduce(T* a, int N, int i) {
  __syncthreads();
  for (N=N/2; N>0; N/=2) {
    if (i < N) a[i] += a[N+i];
    __syncthreads();
  }
}


template <class T>
__device__ T sumKernelLoop(T *x, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    a += x[i];
  return a;
}


template <class T>
__global__ void sumKernel(T *a, T *x, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM];

  cache[t] = sumKernelLoop(x, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}




// SUM-AT (CUDA)
// -------------

template <class T>
__device__ T sumAtKernelLoop(T *x, int *is, int IS, int i, int DI) {
  T a = T();
  for (; i<IS; i+=DI)
    a += x[is[i]];
  return a;
}


template <class T>
__global__ void sumAtKernel(T *a, T *x, T *is, int IS) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM];

  cache[t] = sumAtKernelLoop(x, is, IS, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}




// SUM-IF-NOT (CUDA)
// -----------------

template <class T, class C>
__device__ T sumIfNotKernelLoop(T *x, C *cs, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    if (!cs[i]) a += x[i];
  return a;
}


template <class T, class C>
__global__ void sumIfNotKernel(T *a, T *x, C *cs, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM];

  cache[t] = sumIfNotKernelLoop(x, cs, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}




// L1-NORM (CUDA)
// --------------

template <class T>
__device__ T l1NormKernelLoop(T *x, T *y, int N, int i, int DI) {
  T a = T();
  for (; i<N; i+=DI)
    a += abs(x[i] - y[i]);
  return a;
}


template <class T>
__global__ void l1NormKernel(T *a, T *x, T *y, int N) {
  DEFINE(t, b, B, G);
  __shared__ T cache[BLOCK_DIM];

  cache[t] = l1NormKernelLoop(x, y, N, B*b+t, G*B);
  sumKernelReduce(cache, B, t);
  if (t == 0) a[b] = cache[0];
}




// MULTIPLY (CUDA)
// ---------------

template <class T>
__device__ void multiplyKernelLoop(T *a, T *x, T *y, int N, int i, int DI) {
  for (; i<N; i+=DI)
    a[i] = x[i] * y[i];
}


template <class T>
__global__ void multiplyKernel(T *a, T *x, T* y, int N) {
  DEFINE(t, b, B, G);

  multiplyKernelLoop(a, x, y, N, B*b+t, G*B);
}
