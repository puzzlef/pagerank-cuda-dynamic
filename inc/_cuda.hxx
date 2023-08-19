#pragma once
#include <utility>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <ostream>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include "_debug.hxx"
#include "_cmath.hxx"
#include "_iostream.hxx"

using std::pair;
using std::vector;
using std::make_pair;
using std::min;
using std::max;
using std::abs;
using std::fprintf;
using std::exit;




// LAUNCH CONFIG
// -------------

// Limits
#define BLOCK_LIMIT        1024
#define BLOCK_LIMIT_MAP    256
#define BLOCK_LIMIT_REDUCE 256

#define GRID_LIMIT         65535
#define GRID_LIMIT_MAP     GRID_LIMIT
#define GRID_LIMIT_REDUCE  1024


// Sizes
inline int BLOCK_SIZE(size_t N, int BLIM) noexcept {
  return int(min(N, size_t(BLIM)));
}
inline int GRID_SIZE(size_t N, int B, int GLIM) noexcept {
  return int(min(ceilDiv(N, size_t(B)), size_t(GLIM)));
}




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

#define TRY_CUDA(exp)  tryCuda(exp, #exp, __func__, __LINE__, __FILE__)
#define TRY_CUDAE(exp) PERFORME(TRY_CUDA(exp))
#define TRY_CUDAW(exp) PERFORMW(TRY_CUDA(exp))
#define TRY_CUDAI(exp) PERFORMI(TRY_CUDA(exp))
#define TRY_CUDAD(exp) PERFORMD(TRY_CUDA(exp))
#define TRY_CUDAT(exp) PERFORMT(TRY_CUDA(exp))
#endif

#ifndef TRY
#define TRY(exp)  TRY_CUDA(exp)
#define TRYE(exp) TRY_CUDAE(exp)
#define TRYW(exp) TRY_CUDAW(exp)
#define TRYI(exp) TRY_CUDAI(exp)
#define TRYD(exp) TRY_CUDAD(exp)
#define TRYT(exp) TRY_CUDAT(exp)
#endif

#ifndef ASSERT_CUDA_KERNEL
#define ASSERT_CUDA_KERNEL() TRYD( cudaDeviceSynchronize() )
#endif

#ifndef ASSERT_KERNEL
#define ASSERT_KERNEL() ASSERT_CUDA_KERNEL()
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
#define UNUSED_CUDA(x) unusedCuda(x)
#endif

#ifndef UNUSED
#define UNUSED UNUSED_CUDA
#endif




// REMOVE IDE SQUIGGLES
// --------------------

#ifndef __global__
#define __global__
#define __host__
#define __device__
#define __shared__
void __syncthreads();
#endif




// DEVICE-VALUE(S)/PAIR(S)
// -----------------------

template <class T>
struct DeviceValue {
  const T *data;
  DeviceValue(const T *data)
  : data(data) {}
};

template <class T>
struct DeviceValues {
  const T *data;
  const size_t size;
  DeviceValues(const T *data, size_t size)
  : data(data), size(size) {}
};


template <class K, class V>
struct DevicePair {
  const K *key;
  const V *value;
  DevicePair(const K *key, const V *value)
  : key(key), value(value) {}
};

template <class K, class V>
struct DevicePairs {
  const K *keys;
  const V *values;
  const size_t size;
  DevicePairs(const K *keys, const V *values, size_t size)
  : keys(keys), values(values), size(size) {}
};




// READ
// ----

template <class T>
T readValueCu(const T *v) {
  ASSERT(v);
  T vH;
  TRY( cudaMemcpy(&vH, v, sizeof(T), cudaMemcpyDeviceToHost) );
  return vH;
}

template <class K, class V>
pair<K, V> readPairCu(const K *k, const V *v) {
  ASSERT(k && v);
  K kH; V vH;
  TRY( cudaMemcpy(&kH, k, sizeof(K), cudaMemcpyDeviceToHost) );
  TRY( cudaMemcpy(&vH, v, sizeof(V), cudaMemcpyDeviceToHost) );
  return make_pair(kH, vH);
}

template <class T>
vector<T> readValuesCu(const T *x, size_t N) {
  ASSERT(x);
  vector<T> xH(N);
  size_t N1 = N * sizeof(T);
  TRY( cudaMemcpy(xH.data(), x, N1, cudaMemcpyDeviceToHost) );
  return xH;
}

template <class K, class V>
vector<pair<K, V>> readPairsCu(const K *xk, const V *xv, size_t N) {
  ASSERT(xk && xv);
  vector<K> xkH(N); vector<V> xvH(N);
  vector<pair<K, V>> xH(N);
  size_t NK1 = N * sizeof(K);
  size_t NV1 = N * sizeof(V);
  TRY( cudaMemcpy(xkH.data(), xk, NK1, cudaMemcpyDeviceToHost) );
  TRY( cudaMemcpy(xvH.data(), xv, NV1, cudaMemcpyDeviceToHost) );
  for (size_t i=0; i<N; ++i)
    xH[i] = make_pair(xkH[i], xvH[i]);
  return xH;
}


template <class T>
inline T readCu(const T *v) {
  ASSERT(v);
  return readValueCu(v);
}
template <class K, class V>
inline pair<K, V> readCu(const K *k, const V *v) {
  ASSERT(k && v);
  return readPairCu(k, v);
}
template <class T>
inline vector<T> readCu(const T *x, size_t N) {
  ASSERT(x);
  return readValuesCu(x, N);
}
template <class K, class V>
inline vector<pair<K, V>> readCu(const K *xk, const V *xv, size_t N) {
  ASSERT(xk && xv);
  return readPairsCu(xk, xv, N);
}


template <class T>
inline T read(const DeviceValue<T>& v) {
  return readCu(v.data);
}
template <class T>
inline vector<T> read(const DeviceValues<T>& x) {
  return readCu(x.data, x.size);
}
template <class K, class V>
inline pair<K, V> read(const DevicePair<K, V>& v) {
  return readCu(v.key, v.value);
}
template <class K, class V>
inline vector<pair<K, V>> read(const DevicePairs<K, V>& x) {
  return readCu(x.keys, x.values, x.size);
}




// REDUCE
// ------

inline int reduceSizeCu(size_t N) noexcept {
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  return G;
}




// SWAP
// ----

template <class T>
__device__ void swapDev(T& x, T& y) {
  T t = x; x = y; y = t;
}




// CEIL-DIV
// --------

template <class T>
__device__ T ceilDivDev(T x, T y) {
  ASSERT(y);
  return (x + y-1) / y;
}
template <>
__device__ float ceilDivDev<float>(float x, float y)     { return ceil(x/y); }
template <>
__device__ double ceilDivDev<double>(double x, double y) { return ceil(x/y); }




// COPY
// ----

template <class T>
__device__ void copyKernelLoopW(T *a, const T *x, size_t N, size_t i, size_t DI) {
  ASSERT(a && x && DI);
  for (; i<N; i+=DI)
    a[i] = x[i];
}

template <class T>
__global__ void copyKernelW(T *a, const T *x, size_t N) {
  DEFINE(t, b, B, G);
  ASSERT(a && x);
  copyKernelLoopW(a, x, N, B*b+t, G*B);
}

template <class T>
void copyCuW(T *a, const T *x, size_t N) {
  ASSERT(a && x);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_MAP);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_MAP);
  copyKernelW<<<G, B>>>(a, x, N);
  ASSERT_KERNEL();
}




// GET-ALL
// -------

template <class K, class T>
__device__ void getAllKernelLoopW(T *a, const T *x, const K *is, size_t IS, size_t i, size_t DI) {
  ASSERT(a && x && is && DI);
  for (; i<IS; i+=DI)
    a[i] = x[is[i]];
}

template <class K, class T>
__global__ void getAllKernelW(T *a, const T *x, const K *is, size_t IS) {
  DEFINE(t, b, B, G);
  ASSERT(a && x && is);
  getAllKernelLoopW(a, x, is, IS, B*b+t, G*B);
}

template <class K, class T>
void getAllCuW(T *a, const T *x, const K *is, size_t IS) {
  ASSERT(a && x && is);
  const int B = BLOCK_SIZE(IS,  BLOCK_LIMIT_MAP);
  const int G = GRID_SIZE(IS, B, GRID_LIMIT_MAP);
  getAllKernelW<<<G, B>>>(a, x, is, IS);
  ASSERT_KERNEL();
}




// FILL
// ----

template <class T>
__device__ void fillKernelLoopU(T *a, size_t N, T v, size_t i, size_t DI) {
  ASSERT(a && DI);
  for (; i<N; i+=DI)
    a[i] = v;
}

template <class T>
__global__ void fillKernelU(T *a, size_t N, T v) {
  DEFINE(t, b, B, G);
  ASSERT(a);
  fillKernelLoopU(a, N, v, B*b+t, G*B);
}

template <class T>
void fillCuU(T *a, size_t N, T v) {
  ASSERT(a);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_MAP);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_MAP);
  fillKernelU<<<G, B>>>(a, N, v);
  ASSERT_KERNEL();
}




// FILL-AT
// -------

template <class T, class K>
__device__ void fillAtKernelLoopU(T *a, T v, const K *is, size_t IS, size_t i, size_t DI) {
  ASSERT(a && is && DI);
  for (; i<IS; i+=DI)
    a[is[i]] = v;
}

template <class T, class K>
__global__ void fillAtKernelU(T *a, T v, const K *is, size_t IS) {
  DEFINE(t, b, B, G);
  ASSERT(a && is);
  fillAtKernelLoopU(a, v, is, IS, B*b+t, G*B);
}

template <class T, class K>
void fillAtCuU(T *a, T v, const K *is, size_t IS) {
  ASSERT(a && is);
  const int B = BLOCK_SIZE(IS,  BLOCK_LIMIT_MAP);
  const int G = GRID_SIZE(IS, B, GRID_LIMIT_MAP);
  fillAtKernelU<<<G, B>>>(a, v, is, IS);
  ASSERT_KERNEL();
}




// MULTIPLY
// --------

template <class T>
__device__ void multiplyKernelLoopW(T *a, const T *x, const T *y, size_t N, size_t i, size_t DI) {
  ASSERT(a && x && y && DI);
  for (; i<N; i+=DI)
    a[i] = x[i] * y[i];
}

template <class T>
__global__ void multiplyKernelW(T *a, const T *x, const T* y, size_t N) {
  DEFINE(t, b, B, G);
  ASSERT(a && x && y);
  multiplyKernelLoopW(a, x, y, N, B*b+t, G*B);
}

template <class T>
void multiplyCuW(T *a, const T *x, const T* y, size_t N) {
  ASSERT(a && x && y);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_MAP);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_MAP);
  multiplyKernelW<<<G, B>>>(a, x, y, N);
  ASSERT_KERNEL();
}




// MULTIPLY-VALUE
// --------------

template <class T>
__device__ void multiplyValueKernelLoopW(T *a, const T *x, T v, size_t N, size_t i, size_t DI) {
  ASSERT(a && x && DI);
  for (; i<N; i+=DI)
    a[i] = x[i] * v;
}

template <class T>
__global__ void multiplyValueKernelW(T *a, const T *x, T v, size_t N) {
  DEFINE(t, b, B, G);
  ASSERT(a && x);
  multiplyValueKernelLoopW(a, x, v, N, B*b+t, G*B);
}

template <class T>
void multiplyValueCuW(T *a, const T *x, T v, size_t N) {
  ASSERT(a && x);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_MAP);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_MAP);
  multiplyValueKernelW<<<G, B>>>(a, x, v, N);
  ASSERT_KERNEL();
}




// MAX
// ---

template <class T>
__device__ void maxValueReduceDevU(T* a, size_t N, size_t i) {
  ASSERT(a && N);
  __syncthreads();
  for (; N>1;) {
    size_t DN = (N+1)/2;
    if (i<N/2 && a[DN+i]>a[i]) a[i] = a[DN+i];
    __syncthreads();
    N = DN;
  }
}

template <class K, class V>
__device__ void maxPairReduceDevU(K *ak, V *av, size_t N, size_t i) {
  ASSERT(ak && av && N);
  __syncthreads();
  for (; N>1;) {
    size_t DN = (N+1)/2;
    if (i<N/2 && av[DN+i]>av[i]) {
      ak[i] = ak[DN+i];
      av[i] = av[DN+i];
    }
    __syncthreads();
    N = DN;
  }
}

// Get pair with maximum value, but minimum key.
template <class K, class V>
__device__ void maxPairMinKeyReduceDevU(K *ak, V *av, size_t N, size_t i) {
  ASSERT(ak && av && N);
  __syncthreads();
  for (; N>1;) {
    size_t DN = (N+1)/2;
    if (i<N/2 && (av[DN+i]>av[i] || (av[DN+i]==av[i] && ak[DN+i]<ak[i]))) {
      ak[i] = ak[DN+i];
      av[i] = av[DN+i];
    }
    __syncthreads();
    N = DN;
  }
}

template <class T>
__device__ T maxValueDev(const T *x, size_t N, size_t i, size_t DI) {
  ASSERT(x && DI);
  T a = T();
  for (; i<N; i+=DI)
    a = max(a, x[i]);
  return a;
}

template <class T, int S=BLOCK_LIMIT_REDUCE>
__global__ void maxValueKernelW(T *a, const T *x, size_t N) {
  DEFINE(t, b, B, G);
  ASSERT(a && x);
  __shared__ T cache[S];
  cache[t] = maxValueDev(x, N, B*b+t, G*B);
  maxValueReduceDevU(cache, B, t);
  if (t==0) a[b] = cache[0];
}

template <class T>
void maxValueMemcpyCuW(T *a, const T *x, size_t N) {
  ASSERT(a && x);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  maxValueKernelW<<<G, B>>>(a, x, N);
  ASSERT_KERNEL();
}

template <class T>
void maxValueInplaceCuW(T *a, const T *x, size_t N) {
  ASSERT(a && x);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  maxValueKernelW<<<G, B>>>(a, x, N);
  ASSERT_KERNEL();
  maxValueKernelW<<<1, G>>>(a, a, G);
  ASSERT_KERNEL();
}

template <class T>
void maxValueCuW(T *a, const T *x, size_t N) {
  ASSERT(a && x);
  maxValueMemcpyCuW(a, x, N);
}




// SUM
// ---

template <class T>
__device__ void sumValuesReduceDevU(T* a, size_t N, size_t i) {
  ASSERT(a && N);
  __syncthreads();
  for (; N>1;) {
    size_t DN = (N+1)/2;
    if (i<N/2) a[i] += a[DN+i];
    __syncthreads();
    N = DN;
  }
}

template <class T>
__device__ T sumValuesDev(const T *x, size_t N, size_t i, size_t DI) {
  ASSERT(x && DI);
  T a = T();
  for (; i<N; i+=DI)
    a += x[i];
  return a;
}

template <class T, int S=BLOCK_LIMIT_REDUCE>
__global__ void sumValuesKernelW(T *a, const T *x, size_t N) {
  DEFINE(t, b, B, G);
  ASSERT(a && x);
  __shared__ T cache[S];
  cache[t] = sumValuesDev(x, N, B*b+t, G*B);
  sumValuesReduceDevU(cache, B, t);
  if (t==0) a[b] = cache[0];
}

template <class T>
void sumValuesMemcpyCuW(T *a, const T *x, size_t N) {
  ASSERT(a && x);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  sumValuesKernelW<<<G, B>>>(a, x, N);
  ASSERT_KERNEL();
}

template <class T>
void sumValuesInplaceCuW(T *a, const T *x, size_t N) {
  ASSERT(a && x);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  sumValuesKernelW<<<G, B>>>(a, x, N);
  ASSERT_KERNEL();
  sumValuesKernelW<<<1, G>>>(a, a, G);
  ASSERT_KERNEL();
}

template <class T>
void sumValuesCuW(T *a, const T *x, size_t N) {
  ASSERT(a && x);
  sumValuesMemcpyCuW(a, x, N);
}




// SUM-ABS
// -------

template <class T>
__device__ T sumAbsDev(const T *x, size_t N, size_t i, size_t DI) {
  ASSERT(x && DI);
  T a = T();
  for (; i<N; i+=DI)
    a += abs(x[i]);
  return a;
}

template <class T, int S=BLOCK_LIMIT_REDUCE>
__global__ void sumAbsKernelW(T *a, const T *x, size_t N) {
  DEFINE(t, b, B, G);
  ASSERT(a && x);
  __shared__ T cache[S];
  cache[t] = sumAbsDev(x, N, B*b+t, G*B);
  sumValuesReduceDevU(cache, B, t);
  if (t==0) a[b] = cache[0];
}

template <class T>
void sumAbsMemcpyCuW(T *a, const T *x, size_t N) {
  ASSERT(a && x);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  sumAbsKernelW<<<G, B>>>(a, x, N);
  ASSERT_KERNEL();
}

template <class T>
void sumAbsInplaceCuW(T *a, const T *x, size_t N) {
  ASSERT(a && x);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  sumAbsKernelW<<<G, B>>>(a, x, N);
  ASSERT_KERNEL();
  sumValuesKernelW<<<1, G>>>(a, a, G);
  ASSERT_KERNEL();
}

template <class T>
void sumAbsCuW(T *a, const T *x, size_t N) {
  ASSERT(a && x);
  sumAbsMemcpyCuW(a, x, N);
}




// SUM-SQR
// -------

template <class T>
__device__ T sumSqrDev(const T *x, size_t N, size_t i, size_t DI) {
  ASSERT(x && DI);
  T a = T();
  for (; i<N; i+=DI)
    a += x[i]*x[i];
  return a;
}

template <class T, int S=BLOCK_LIMIT_REDUCE>
__global__ void sumSqrKernelW(T *a, const T *x, size_t N) {
  DEFINE(t, b, B, G);
  ASSERT(a && x);
  __shared__ T cache[S];
  cache[t] = sumSqrDev(x, N, B*b+t, G*B);
  sumValuesReduceDevU(cache, B, t);
  if (t==0) a[b] = cache[0];
}

template <class T>
void sumSqrMemcpyCuW(T *a, const T *x, size_t N) {
  ASSERT(a && x);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  sumSqrKernelW<<<G, B>>>(a, x, N);
  ASSERT_KERNEL();
}

template <class T>
void sumSqrInplaceCuW(T *a, const T *x, size_t N) {
  ASSERT(a && x);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  sumSqrKernelW<<<G, B>>>(a, x, N);
  ASSERT_KERNEL();
  sumValuesKernelW<<<1, G>>>(a, a, G);
  ASSERT_KERNEL();
}

template <class T>
void sumSqrCuW(T *a, const T *x, size_t N) {
  ASSERT(a && x);
  sumSqrMemcpyCuW(a, x, N);
}




// SUM-AT
// ------

template <class T, class K>
__device__ T sumAtDev(const T *x, const K *is, size_t IS, size_t i, size_t DI) {
  ASSERT(x && is && DI);
  T a = T();
  for (; i<IS; i+=DI)
    a += x[is[i]];
  return a;
}

template <class T, class K, int S=BLOCK_LIMIT_REDUCE>
__global__ void sumAtKernelW(T *a, const T *x, const K *is, size_t IS) {
  DEFINE(t, b, B, G);
  ASSERT(a && x && is);
  __shared__ T cache[S];
  cache[t] = sumAtDev(x, is, IS, B*b+t, G*B);
  sumValuesReduceDevU(cache, B, t);
  if (t==0) a[b] = cache[0];
}

template <class T, class K>
void sumAtMemcpyCuW(T *a, const T *x, const K *is, size_t IS) {
  ASSERT(a && x && is);
  const int B = BLOCK_SIZE(IS,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(IS, B, GRID_LIMIT_REDUCE);
  sumAtKernelW<<<G, B>>>(a, x, is, IS);
  ASSERT_KERNEL();
}

template <class T, class K>
void sumAtInplaceCuW(T *a, const T *x, const K *is, size_t IS) {
  ASSERT(a && x && is);
  const int B = BLOCK_SIZE(IS,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(IS, B, GRID_LIMIT_REDUCE);
  sumAtKernelW<<<G, B>>>(a, x, is, IS);
  ASSERT_KERNEL();
  sumValuesKernelW<<<1, G>>>(a, a, G);
  ASSERT_KERNEL();
}

template <class T, class K>
void sumAtCuW(T *a, const T *x, const K *is, size_t IS) {
  ASSERT(a && x && is);
  sumAtMemcpyCuW(a, x, is, IS);
}




// SUM-IF-NOT
// ----------

template <class T, class C>
__device__ T sumIfNotDev(const T *x, const C *cs, size_t N, size_t i, size_t DI) {
  ASSERT(x && cs && DI);
  T a = T();
  for (; i<N; i+=DI)
    if (!cs[i]) a += x[i];
  return a;
}

template <class T, class C, int S=BLOCK_LIMIT_REDUCE>
__global__ void sumIfNotKernelW(T *a, const T *x, const C *cs, size_t N) {
  DEFINE(t, b, B, G);
  ASSERT(a && x && cs);
  __shared__ T cache[S];
  cache[t] = sumIfNotDev(x, cs, N, B*b+t, G*B);
  sumValuesReduceDevU(cache, B, t);
  if (t==0) a[b] = cache[0];
}

template <class T, class C>
void sumIfNotMemcpyCuW(T *a, const T *x, const C *cs, size_t N) {
  ASSERT(a && x && cs);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  sumIfNotKernelW<<<G, B>>>(a, x, cs, N);
  ASSERT_KERNEL();
}

template <class T, class C>
void sumIfNotInplaceCuW(T *a, const T *x, const C *cs, size_t N) {
  ASSERT(a && x && cs);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  sumIfNotKernelW<<<G, B>>>(a, x, cs, N);
  ASSERT_KERNEL();
  sumValuesKernelW<<<1, G>>>(a, a, G);
  ASSERT_KERNEL();
}

template <class T, class C>
void sumIfNotCuW(T *a, const T *x, const C *cs, size_t N) {
  ASSERT(a && x && cs);
  sumIfNotMemcpyCuW(a, x, cs, N);
}




// SUM-MULTIPLY
// ------------

template <class T>
__device__ T sumMultiplyDev(const T *x, const T *y, size_t N, size_t i, size_t DI) {
  ASSERT(x && y && DI);
  T a = T();
  for (; i<N; i+=DI)
    a += x[i] * y[i];
  return a;
}

template <class T, int S=BLOCK_LIMIT_REDUCE>
__global__ void sumMultiplyKernelW(T *a, const T *x, const T *y, size_t N) {
  DEFINE(t, b, B, G);
  ASSERT(a && x && y);
  __shared__ T cache[S];
  cache[t] = sumMultiplyDev(x, y, N, B*b+t, G*B);
  sumValuesReduceDevU(cache, B, t);
  if (t==0) a[b] = cache[0];
}

template <class T>
void sumMultiplyMemcpyCuW(T *a, const T *x, const T *y, size_t N) {
  ASSERT(a && x && y);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  sumMultiplyKernelW<<<G, B>>>(a, x, y, N);
  ASSERT_KERNEL();
}

template <class T>
void sumMultiplyInplaceCuW(T *a, const T *x, const T *y, size_t N) {
  ASSERT(a && x && y);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  sumMultiplyKernelW<<<G, B>>>(a, x, y, N);
  ASSERT_KERNEL();
  sumValuesKernelW<<<1, G>>>(a, a, G);
  ASSERT_KERNEL();
}

template <class T>
void sumMultiplyCuW(T *a, const T *x, const T *y, size_t N) {
  ASSERT(a && x && y);
  sumMultiplyMemcpyCuW(a, x, y, N);
}




// SUM-MULTIPLY-AT
// ---------------

template <class T, class K>
__device__ T sumMultiplyAtDev(const T *x, const T *y, const K *is, size_t IS, size_t i, size_t DI) {
  ASSERT(x && y && is && DI);
  T a = T();
  for (; i<IS; i+=DI)
    a += x[is[i]] * y[is[i]];
  return a;
}

template <class T, class K, int S=BLOCK_LIMIT_REDUCE>
__global__ void sumMultiplyAtKernelW(T *a, const T *x, const T *y, const K *is, size_t IS) {
  DEFINE(t, b, B, G);
  ASSERT(a && x && y && is);
  __shared__ T cache[S];
  cache[t] = sumMultiplyAtDev(x, y, is, IS, B*b+t, G*B);
  sumValuesReduceDevU(cache, B, t);
  if (t==0) a[b] = cache[0];
}

template <class T, class K>
void sumMultiplyAtMemcpyCuW(T *a, const T *x, const T *y, const K *is, size_t IS) {
  ASSERT(a && x && y && is);
  const int B = BLOCK_SIZE(IS,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(IS, B, GRID_LIMIT_REDUCE);
  sumMultiplyAtKernelW<<<G, B>>>(a, x, y, is, IS);
  ASSERT_KERNEL();
}

template <class T, class K>
void sumMultiplyAtInplaceCuW(T *a, const T *x, const T *y, const K *is, size_t IS) {
  ASSERT(a && x && y && is);
  const int B = BLOCK_SIZE(IS,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(IS, B, GRID_LIMIT_REDUCE);
  sumMultiplyAtKernelW<<<G, B>>>(a, x, y, is, IS);
  ASSERT_KERNEL();
  sumValuesKernelW<<<1, G>>>(a, a, G);
  ASSERT_KERNEL();
}

template <class T, class K>
void sumMultiplyAtCuW(T *a, const T *x, const T *y, const K *is, size_t IS) {
  ASSERT(a && x && y && is);
  sumMultiplyAtMemcpyCuW(a, x, y, is, IS);
}




// COUNT
// -----

template <class T>
__device__ size_t countValueDev(const T *x, T v, size_t N, size_t i, size_t DI) {
  ASSERT(x && DI);
  size_t a = 0;
  for (; i<N; i+=DI)
    if (x[i]==v) ++a;
  return a;
}

template <class T, class TA, int S=BLOCK_LIMIT_REDUCE>
__global__ void countValueKernelW(TA *a, const T *x, T v, size_t N) {
  DEFINE(t, b, B, G);
  ASSERT(a && x);
  __shared__ TA cache[S];
  cache[t] = countValueDev(x, v, N, B*b+t, G*B);
  sumValuesReduceDevU(cache, B, t);
  if (t==0) a[b] = cache[0];
}

template <class T, class TA>
void countValueMemcpyCuW(TA *a, const T *x, T v, size_t N) {
  ASSERT(a && x);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  countValueKernelW<<<G, B>>>(a, x, v, N);
  ASSERT_KERNEL();
}

template <class T, class TA>
void countValueInplaceCuW(TA *a, const T *x, T v, size_t N) {
  ASSERT(a && x);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  countValueKernelW<<<G, B>>>(a, x, v, N);
  ASSERT_KERNEL();
  countValueKernelW<<<1, G>>>(a, a, v, G);
  ASSERT_KERNEL();
}

template <class T, class TA>
void countValueCuW(TA *a, const T *x, T v, size_t N) {
  ASSERT(a && x);
  countValueMemcpyCuW(a, x, v, N);
}




// COUNT-NON-ZERO
// --------------

template <class T>
__device__ size_t countNonZeroDev(const T *x, size_t N, size_t i, size_t DI) {
  ASSERT(x && DI);
  size_t a = 0;
  for (; i<N; i+=DI)
    if (x[i]) ++a;
  return a;
}

template <class T, class TA, int S=BLOCK_LIMIT_REDUCE>
__global__ void countNonZeroKernelW(TA *a, const T *x, size_t N) {
  DEFINE(t, b, B, G);
  ASSERT(a && x);
  __shared__ TA cache[S];
  cache[t] = countNonZeroDev(x, N, B*b+t, G*B);
  sumValuesReduceDevU(cache, B, t);
  if (t==0) a[b] = cache[0];
}

template <class T, class TA>
void countNonZeroMemcpyCuW(TA *a, const T *x, size_t N) {
  ASSERT(a && x);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  countNonZeroKernelW<<<G, B>>>(a, x, N);
  ASSERT_KERNEL();
}

template <class T, class TA>
void countNonZeroInplaceCuW(TA *a, const T *x, size_t N) {
  ASSERT(a && x);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  countNonZeroKernelW<<<G, B>>>(a, x, N);
  ASSERT_KERNEL();
  countNonZeroKernelW<<<1, G>>>(a, a, G);
  ASSERT_KERNEL();
}

template <class T, class TA>
void countNonZeroCuW(TA *a, const T *x, size_t N) {
  ASSERT(a && x);
  countNonZeroMemcpyCuW(a, x, N);
}




// L1-NORM
// -------

template <class T>
__device__ T l1NormDev(const T *x, const T *y, size_t N, size_t i, size_t DI) {
  ASSERT(x && y && DI);
  T a = T();
  for (; i<N; i+=DI)
    a += abs(x[i] - y[i]);
  return a;
}

template <class T, int S=BLOCK_LIMIT_REDUCE>
__global__ void l1NormKernelW(T *a, const T *x, const T *y, size_t N) {
  DEFINE(t, b, B, G);
  ASSERT(a && x && y);
  __shared__ T cache[S];
  cache[t] = l1NormDev(x, y, N, B*b+t, G*B);
  sumValuesReduceDevU(cache, B, t);
  if (t==0) a[b] = cache[0];
}

template <class T>
void l1NormMemcpyCuW(T *a, const T *x, const T *y, size_t N) {
  ASSERT(a && x && y);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  l1NormKernelW<<<G, B>>>(a, x, y, N);
  ASSERT_KERNEL();
}

template <class T>
void l1NormInplaceCuW(T *a, const T *x, const T *y, size_t N) {
  ASSERT(a && x && y);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  l1NormKernelW<<<G, B>>>(a, x, y, N);
  ASSERT_KERNEL();
  sumValuesKernelW<<<1, G>>>(a, a, G);
  ASSERT_KERNEL();
}

template <class T>
void l1NormCuW(T *a, const T *x, const T *y, size_t N) {
  ASSERT(a && x && y);
  l1NormMemcpyCuW(a, x, y, N);
}




// L2-NORM
// -------
// Remember to sqrt the result!

template <class T>
__device__ T l2NormDev(const T *x, const T *y, size_t N, size_t i, size_t DI) {
  ASSERT(x && y && DI);
  T a = T();
  for (; i<N; i+=DI)
    a += (x[i] - y[i]) * (x[i] - y[i]);
  return a;
}

template <class T, int S=BLOCK_LIMIT_REDUCE>
__global__ void l2NormKernelW(T *a, const T *x, const T *y, size_t N) {
  DEFINE(t, b, B, G);
  ASSERT(a && x && y);
  __shared__ T cache[S];
  cache[t] = l2NormDev(x, y, N, B*b+t, G*B);
  sumValuesReduceDevU(cache, B, t);
  if (t==0) a[b] = cache[0];
}

template <class T>
void l2NormMemcpyCuW(T *a, const T *x, const T *y, size_t N) {
  ASSERT(a && x && y);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  l2NormKernelW<<<G, B>>>(a, x, y, N);
  ASSERT_KERNEL();
}

template <class T>
void l2NormInplaceCuW(T *a, const T *x, const T *y, size_t N) {
  ASSERT(a && x && y);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  l2NormKernelW<<<G, B>>>(a, x, y, N);
  ASSERT_KERNEL();
  sumValuesKernelW<<<1, G>>>(a, a, G);
  ASSERT_KERNEL();
}

template <class T>
void l2NormCuW(T *a, const T *x, const T *y, size_t N) {
  ASSERT(a && x && y);
  l2NormMemcpyCuW(a, x, y, N);
}




// LI-NORM
// -------

template <class T>
__device__ T liNormDev(const T *x, const T *y, size_t N, size_t i, size_t DI) {
  ASSERT(x && y && DI);
  T a = T();
  for (; i<N; i+=DI)
    a = max(a, abs(x[i] - y[i]));
  return a;
}

template <class T, int S=BLOCK_LIMIT_REDUCE>
__global__ void liNormKernelW(T *a, const T *x, const T *y, size_t N) {
  DEFINE(t, b, B, G);
  ASSERT(a && x && y);
  __shared__ T cache[S];
  cache[t] = liNormDev(x, y, N, B*b+t, G*B);
  maxValueReduceDevU(cache, B, t);
  if (t==0) a[b] = cache[0];
}

template <class T>
void liNormMemcpyCuW(T *a, const T *x, const T *y, size_t N) {
  ASSERT(a && x && y);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  liNormKernelW<<<G, B>>>(a, x, y, N);
  ASSERT_KERNEL();
}

template <class T>
void liNormInplaceCuW(T *a, const T *x, const T *y, size_t N) {
  ASSERT(a && x && y);
  const int B = BLOCK_SIZE(N,  BLOCK_LIMIT_REDUCE);
  const int G = GRID_SIZE(N, B, GRID_LIMIT_REDUCE);
  liNormKernelW<<<G, B>>>(a, x, y, N);
  ASSERT_KERNEL();
  maxValueKernelW<<<1, G>>>(a, a, G);
  ASSERT_KERNEL();
}

template <class T>
void liNormCuW(T *a, const T *x, const T *y, size_t N) {
  ASSERT(a && x && y);
  liNormMemcpyCuW(a, x, y, N);
}




// SCAN
// ----

template <class T>
void exclusiveScanCuW(T *a, const T *x, size_t N) {
  ASSERT(a && x);
  thrust::device_ptr<T> xD((T*) x), aD(a);
  thrust::exclusive_scan(xD, xD+N, aD);
}

template <class T>
void inclusiveScanCuW(T *a, const T *x, size_t N) {
  ASSERT(a && x);
  thrust::device_ptr<T> xD((T*) x), aD(a);
  thrust::inclusive_scan(xD, xD+N, aD);
}
