#pragma once
#include <cmath>
#include <vector>
#include <algorithm>

using std::vector;
using std::copy;
using std::swap;
using std::abs;
using std::max;
using std::sqrt;




// REORDER
// -------
// Ref: https://stackoverflow.com/a/22183350/1413259

template <class T>
void reorder(vector<T>& x, vector<int> is) {
  for(int i=0, N=x.size(); i<N; i++) {
    while(is[i] != is[is[i]]) {
      swap(x[is[i]], x[is[is[i]]]);
      swap(  is[i],    is[is[i]]);
    }
  }
}




// ERASE
// -----

template <class T>
void eraseIndex(vector<T>& a, int i) {
  a.erase(a.begin()+i);
}

template <class T>
void eraseIndex(vector<T>& a, int i, int I) {
  a.erase(a.begin()+i, a.begin()+I);
}




// APPEND
// ------

template <class T, class I>
void append(vector<T>& a, I ib, I ie) {
  for (auto i=ib; i!=ie; ++i)
    a.push_back(*i);
}

template <class T, class J>
void append(vector<T>& a, J&& vs) {
  append(a, vs.begin(), vs.end());
}




// JOIN
// ----

template <class T, class F>
auto joinIf(const vector<vector<T>>& xs, F fn) {
  vector<vector<T>> a;
  for (const auto& x : xs) {
    auto& b = a.back();
    if (a.empty() || !fn(b, x)) a.push_back(x);
    else b.insert(b.end(), x.begin(), x.end());
  }
  return a;
}

template <class T>
auto joinUntilSize(const vector<vector<T>>& xs, int N) {
  return joinIf(xs, [&](const auto& b, const auto& x) { return b.size()<N; });
}

template <class T>
auto join(const vector<vector<T>>& xs) {
  vector<T> a;
  for (const auto& x : xs)
    a.insert(a.end(), x.begin(), x.end());
  return a;
}




// GATHER
// ------

template <class T, class U, class J>
void gather(T *a, const U *x, J&& is) {
  int j = 0;
  for (int i : is)
    a[j++] = x[i];
}

template <class T, class U, class J>
void gather(vector<T>& a, const vector<U>& x, J&& is) {
  gather(a.data(), x.data(), is);
}




// SCATTER
// -------

template <class T, class U, class J>
void scatter(T *a, const U *x, J&& is) {
  int j = 0;
  for (int i : is)
    a[i] = x[j++];
}

template <class T, class U, class J>
void scatter(vector<T>& a, const vector<U>& x, J&& is) {
  scatter(a.data(), x.data(), is);
}




// COPY
// ----

template <class T, class U>
void copy(T *a, U *x, int N) {
  for (int i=0; i<N; i++)
    a[i] = x[i];
}

template <class T, class U>
void copy(vector<T>& a, const vector<U>& x) {
  copy(a.data(), x.data(), int(x.size()));
}

template <class T, class U>
void copy(vector<T>& a, const vector<T>& x, int i, int N) {
  copy(a.data()+i, x.data()+i, N);
}


template <class T, class U>
void copyOmp(T *a, U *x, int N) {
  #pragma omp parallel for schedule(static,4096)
  for (int i=0; i<N; i++)
    a[i] = x[i];
}

template <class T, class U>
void copyOmp(vector<T>& a, const vector<U>& x) {
  copyOmp(a.data(), x.data(), int(x.size()));
}

template <class T, class U>
void copyOmp(vector<T>& a, const vector<U>& x, int i, int N) {
  copyOmp(a.data()+i, x.data()+i, N);
}




// FILL
// ----

template <class T, class U>
void fill(T *a, int N, const U& v) {
  for (int i=0; i<N; i++)
    a[i] = v;
}

template <class T, class U>
void fill(vector<T>& a, const U& v) {
  fill(a.begin(), a.end(), v);
}

template <class T, class U>
void fill(vector<T>& a, int i, int N, const U& v) {
  fill(a.data()+i, N, v);
}


template <class T, class U>
void fillOmp(T *a, int N, const U& v) {
  #pragma omp parallel for schedule(static,4096)
  for (int i=0; i<N; i++)
    a[i] = v;
}

template <class T, class U>
void fillOmp(vector<T>& a, const U& v) {
  fillOmp(a.data(), int(a.size()), v);
}

template <class T, class U>
void fillOmp(vector<T>& a, int i, int N, const U& v) {
  fillOmp(a.data()+i, N, v);
}




// FILL-AT
// -------

template <class T, class U, class J>
void fillAt(T *a, const U& v, J&& is) {
  for (int i : is)
    a[i] = v;
}

template <class T, class U, class J>
void fillAt(vector<T>& a, const U& v, J&& is) {
  fillAt(a.data(), v, is);
}

template <class T, class U, class J>
void fillAt(vector<T>& a, int i, const U& v, J&& is) {
  fillAt(a.data()+i, v, is);
}




// SUM
// ---

template <class T, class U=T>
U sum(const T *x, int N, U a=U()) {
  for (int i=0; i<N; i++)
    a += x[i];
  return a;
}

template <class T, class U=T>
U sum(const vector<T>& x, U a=U()) {
  return sum(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U sum(const vector<T>& x, int i, int N, U a=U()) {
  return sum(x.data()+i, N, a);
}


template <class T, class U=T>
U sumOmp(const T *x, int N, U a=U()) {
  #pragma omp parallel for schedule(static,4096) reduction(+:a)
  for (int i=0; i<N; i++)
    a += x[i];
  return a;
}

template <class T, class U=T>
U sumOmp(const vector<T>& x, U a=U()) {
  return sumOmp(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U sumOmp(const vector<T>& x, int i, int N, U a=U()) {
  return sumOmp(x.data()+i, N, a);
}




// SUM-ABS
// -------

template <class T, class U=T>
U sumAbs(const T *x, int N, U a=U()) {
  for (int i=0; i<N; i++)
    a += abs(x[i]);
  return a;
}

template <class T, class U=T>
U sumAbs(const vector<T>& x, U a=U()) {
  return sumAbs(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U sumAbs(const vector<T>& x, int i, int N, U a=U()) {
  return sumAbs(x.data()+i, N, a);
}


template <class T, class U=T>
U sumAbsOmp(const T *x, int N, U a=U()) {
  #pragma omp parallel for schedule(static,4096) reduction(+:a)
  for (int i=0; i<N; i++)
    a += abs(x[i]);
  return a;
}

template <class T, class U=T>
U sumAbsOmp(const vector<T>& x, U a=U()) {
  return sumAbsOmp(x.data(), int(x.size()), a);
}

template <class T, class U=T>
U sumAbsOmp(const vector<T>& x, int i, int N, U a=U()) {
  return sumAbsOmp(x.data()+i, N, a);
}




// SUM-AT
// ------

template <class T, class J, class U=T>
U sumAt(const T *x, J&& is, U a=U()) {
  for (int i : is)
    a += x[i];
  return a;
}

template <class T, class J, class U=T>
U sumAt(const vector<T>& x, J&& is, U a=U()) {
  return sumAt(x.data(), is, a);
}

template <class T, class J, class U=T>
U sumAt(const vector<T>& x, int i, J&& is, U a=U()) {
  return sumAt(x.data()+i, is, a);
}




// ADD-VALUE
// ---------

template <class T, class U>
void addValue(T *a, int N, const U& v) {
  for (int i=0; i<N; i++)
    a[i] += v;
}

template <class T, class U>
void addValue(vector<T>& a, const U& v) {
  addValue(a.data(), int(a.size()), v);
}

template <class T, class U>
void addValue(vector<T>& a, int i, int N, const U& v) {
  addValue(a.data()+i, N, v);
}


template <class T, class U>
void addValueOmp(T *a, int N, const U& v) {
  #pragma omp parallel for schedule(static,4096)
  for (int i=0; i<N; i++)
    a[i] += v;
}

template <class T, class U>
void addValueOmp(vector<T>& a, const U& v) {
  addValueOmp(a.data(), int(a.size()), v);
}

template <class T, class U>
void addValueOmp(vector<T>& a, int i, int N, const U& v) {
  addValueOmp(a.data()+i, N, v);
}




// ADD-VALUE-AT
// ------------

template <class T, class U, class J>
void addValueAt(T *a, const U& v, J&& is) {
  for (int i : is)
    a[i] += v;
}

template <class T, class U, class J>
void addValueAt(vector<T>& a, const U& v, J&& is) {
  addValueAt(a.data(), v, is);
}

template <class T, class U, class J>
void addValueAt(vector<T>& a, int i, const U& v, J&& is) {
  addValueAt(a.data()+i, v, is);
}




// L1-NORM
// -------

template <class T, class U, class V=T>
V l1Norm(const T *x, const U *y, int N, V a=V()) {
  for (int i=0; i<N; i++)
    a += abs(x[i] - y[i]);
  return a;
}

template <class T, class U, class V=T>
V l1Norm(const vector<T>& x, const vector<U>& y, V a=V()) {
  return l1Norm(x.data(), y.data(), int(x.size()), a);
}

template <class T, class U, class V=T>
V l1Norm(const vector<T>& x, const vector<U>& y, int i, int N, V a=V()) {
  return l1Norm(x.data()+i, y.data()+i, N, a);
}


template <class T, class U, class V=T>
V l1NormOmp(const T *x, const U *y, int N, V a=V()) {
  #pragma omp parallel for schedule(static,4096) reduction(+:a)
  for (int i=0; i<N; i++)
    a += abs(x[i] - y[i]);
  return a;
}

template <class T, class U, class V=T>
V l1NormOmp(const vector<T>& x, const vector<U>& y, V a=V()) {
  return l1NormOmp(x.data(), y.data(), int(x.size()), a);
}

template <class T, class U, class V=T>
V l1NormOmp(const vector<T>& x, const vector<U>& y, int i, int N, V a=V()) {
  return l1NormOmp(x.data()+i, y.data()+i, N, a);
}




// L2-NORM
// -------

template <class T, class U, class V=T>
V l2Norm(const T *x, const U *y, int N, V a=V()) {
  for (int i=0; i<N; i++)
    a += x[i]*x[i] - y[i]*y[i];
  return sqrt(a);
}

template <class T, class U, class V=T>
V l2Norm(const vector<T>& x, const vector<U>& y, V a=V()) {
  return l2Norm(x.data(), y.data(), int(x.size()), a);
}

template <class T, class U, class V=T>
V l2Norm(const vector<T>& x, const vector<U>& y, int i, int N, V a=V()) {
  return l2Norm(x.data()+i, y.data()+i, N, a);
}


template <class T, class U, class V=T>
V l2NormOmp(const T *x, const U *y, int N, V a=V()) {
  #pragma omp parallel for schedule(static,4096) reduction(+:a)
  for (int i=0; i<N; i++)
    a += x[i]*x[i] - y[i]*y[i];
  return sqrt(a);
}

template <class T, class U, class V=T>
V l2NormOmp(const vector<T>& x, const vector<U>& y, V a=V()) {
  return l2NormOmp(x.data(), y.data(), int(x.size()), a);
}

template <class T, class U, class V=T>
V l2NormOmp(const vector<T>& x, const vector<U>& y, int i, int N, V a=V()) {
  return l2NormOmp(x.data()+i, y.data()+i, N, a);
}




// LI-NORM (INFINITY)
// ------------------

template <class T, class U, class V=T>
V liNorm(const T *x, const U *y, int N, V a=V()) {
  for (int i=0; i<N; i++)
    a = max(a, abs(x[i] - y[i]));
  return a;
}

template <class T, class U, class V=T>
V liNorm(const vector<T>& x, const vector<U>& y, V a=V()) {
  return liNorm(x.data(), y.data(), int(x.size()), a);
}

template <class T, class U, class V=T>
V liNorm(const vector<T>& x, const vector<U>& y, int i, int N, V a=V()) {
  return liNorm(x.data()+i, y.data()+i, N, a);
}


template <class T, class U, class V=T>
V liNormOmp(const T *x, const U *y, int N, V a=V()) {
  #pragma omp parallel for schedule(static,4096) reduction(+:a)
  for (int i=0; i<N; i++)
    a = max(a, abs(x[i] - y[i]));
  return a;
}

template <class T, class U, class V=T>
V liNormOmp(const vector<T>& x, const vector<U>& y, V a=V()) {
  return liNormOmp(x.data(), y.data(), int(x.size()), a);
}

template <class T, class U, class V=T>
V liNormOmp(const vector<T>& x, const vector<U>& y, int i, int N, V a=V()) {
  return liNormOmp(x.data()+i, y.data()+i, N, a);
}




// MULTIPLY
// --------

template <class T, class U, class V>
void multiply(T *a, const U *x, const V *y, int N) {
  for (int i=0; i<N; i++)
    a[i] = x[i] * y[i];
}

template <class T, class U, class V>
void multiply(vector<T>& a, const vector<U>& x, const vector<V>& y) {
  multiply(a.data(), x.data(), y.data(), int(x.size()));
}

template <class T, class U, class V>
void multiply(vector<T>& a, const vector<U>& x, const vector<V>& y, int i, int N) {
  multiply(a.data()+i, x.data()+i, y.data()+i, N);
}


template <class T, class U, class V>
void multiplyOmp(T *a, const U *x, const V *y, int N) {
  #pragma omp parallel for schedule(static,4096)
  for (int i=0; i<N; i++)
    a[i] = x[i] * y[i];
}

template <class T, class U, class V>
void multiplyOmp(vector<T>& a, const vector<U>& x, const vector<V>& y) {
  multiplyOmp(a.data(), x.data(), y.data(), int(x.size()));
}

template <class T, class U, class V>
void multiplyOmp(vector<T>& a, const vector<U>& x, const vector<V>& y, int i, int N) {
  multiplyOmp(a.data()+i, x.data()+i, y.data()+i, N);
}
