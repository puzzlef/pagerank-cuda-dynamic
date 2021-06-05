#pragma once
#include <iterator>
#include <unordered_map>
#include <algorithm>

using std::forward_iterator_tag;
using std::random_access_iterator_tag;
using std::unordered_map;
using std::distance;
using std::max;




// ITERATOR-*
// ----------
// Helps create iterators.

#ifndef ITERATOR_USING
#define ITERATOR_USING(cat, dif, val, ref, ptr) \
  using iterator_category = cat; \
  using difference_type   = dif; \
  using value_type = val; \
  using reference  = ref; \
  using pointer    = ptr;

#define ITERATOR_USING_I(I) \
  using iterator_category = typename I::iterator_category; \
  using difference_type   = typename I::difference_type; \
  using value_type = typename I::value_type; \
  using reference  = typename I::reference; \
  using pointer    = typename I::pointer;

#define ITERATOR_USING_IVR(I, val, ref) \
  using iterator_category = typename I::iterator_category; \
  using difference_type   = typename I::difference_type; \
  using value_type = val; \
  using reference  = ref; \
  using pointer    = value_type*;
#endif


#ifndef ITERATOR_DEREF
#define ITERATOR_DEREF(I, se, be, ae) \
  reference operator*() { return se; } \
  reference operator[](difference_type i) { return be; } \
  pointer operator->() { return ae; }
#endif


#ifndef ITERATOR_NEXT
#define ITERATOR_NEXTP(I, ie)  \
  I& operator++() { ie; return *this; }  \
  I operator++(int) { I a = *this; ++(*this); return a; }

#define ITERATOR_NEXTN(I, de) \
  I& operator--() { de; return *this; }  \
  I operator--(int) { I a = *this; --(*this); return a; }

#define ITERATOR_NEXT(I, ie, de) \
  ITERATOR_NEXTP(I, ie) \
  ITERATOR_NEXTN(I, de)
#endif


#ifndef ITERATOR_ADVANCE
#define ITERATOR_ADVANCEP(I, i, fe) \
  I& operator+=(difference_type i) { fe; return *this; }

#define ITERATOR_ADVANCEN(I, i, be) \
  I& operator-=(difference_type i) { be; return *this; }

#define ITERATOR_ADVANCE(I, i, fe, be) \
  ITERATOR_ADVANCEP(I, i, fe) \
  ITERATOR_ADVANCEN(I, i, be)
#endif


#ifndef ITERATOR_ARITHMETICP
#define ITERATOR_ARITHMETICP(I, a, b, ...)  \
  friend I operator+(const I& a, difference_type b) { return I(__VA_ARGS__); } \
  friend I operator+(difference_type b, const I& a) { return I(__VA_ARGS__); }
#endif


#ifndef ITERATOR_ARITHMETICN
#define ITERATOR_ARITHMETICN(I, a, b, ...) \
  friend I operator-(const I& a, difference_type b) { return I(__VA_ARGS__); } \
  friend I operator-(difference_type b, const I& a) { return I(__VA_ARGS__); }
#endif


#ifndef ITERATOR_COMPARISION
#define ITERATOR_COMPARISION(I, a, b, ae, be)  \
  friend bool operator==(const I& a, const I& b) { return ae == be; } \
  friend bool operator!=(const I& a, const I& b) { return ae != be; } \
  friend bool operator>=(const I& a, const I& b) { return ae >= be; } \
  friend bool operator<=(const I& a, const I& b) { return ae <= be; } \
  friend bool operator>(const I& a, const I& b) { return ae > be; } \
  friend bool operator<(const I& a, const I& b) { return ae < be; }
#endif


#ifndef ITERABLE_SIZE
#define ITERABLE_SIZE(se) \
  size_t size() { return se; } \
  bool empty() { return size() == 0; }
#endif




// ITERABLE
// --------

template <class I>
class Iterable {
  const I ib, ie;

  public:
  Iterable(I ib, I ie) : ib(ib), ie(ie) {}
  auto begin() const { return ib; }
  auto end() const   { return ie; }
};


template <class I>
auto iterable(I ib, I ie) {
  return Iterable<I>(ib, ie);
}

template <class J>
auto iterable(const J& x) {
  using I = decltype(x.begin());
  return Iterable<I>(x.begin(), x.end());
}




// SIZED-ITERABLE
// --------------

template <class I>
class SizedIterable : public Iterable<I> {
  const size_t N;

  public:
  SizedIterable(I ib, I ie, size_t N) : Iterable<I>(ib, ie), N(N) {}
  ITERABLE_SIZE(N)
};


template <class I>
auto sizedIterable(I ib, I ie, int N) {
  return SizedIterable<I>(ib, ie, N);
}

template <class I>
auto sizedIterable(I ib, I ie) {
  return SizedIterable<I>(ib, ie, distance(ib, ie));
}

template <class J>
auto sizedIterable(const J& x, int N) {
  using I = decltype(x.begin());
  return Iterable<I>(x.begin(), x.end(), N);
}

template <class J>
auto sizedIterable(const J& x) {
  using I = decltype(x.begin());
  return Iterable<I>(x.begin(), x.end());
}




// SIZE
// ----

template <class T>
int size(const vector<T>& x) {
  return x.size();
}

template <class I>
int size(const SizedIterable<I>& x) {
  return x.size();
}

template <class J>
int size(const J& x) {
  return distance(x.begin(), x.end());
}




// CSIZE
// -----
// Compile-time size.

template <class T>
int csize(const vector<T>& x) {
  return x.size();
}

template <class I>
int csize(const SizedIterable<I>& x) {
  return x.size();
}

template <class J>
int csize(const J& x) {
  return -1;
}




// SLICE
// -----

template <class J>
auto slice(const J& x, int i) {
  auto ib = x.begin(), ie = x.end();
  return sizedIterable(ib+i<ie? ib+i : ie, ie);
}

template <class J>
auto slice(const J& x, int i, int I) {
  auto ib = x.begin(), ie = x.end();
  return sizedIterable(ib+i<ie? ib+i : ie, ib+I<ie? ib+I : ie, I-i);
}




// TRANSFORM
// ---------

template <class I, class F>
class TransformIterator {
  I it;
  const F fn;

  public:
  ITERATOR_USING_IVR(I, decltype(fn(*it)), value_type)
  TransformIterator(I it, F fn) : it(it), fn(fn) {}
  ITERATOR_DEREF(TransformIterator, fn(*it), fn(it[i]), NULL)
  ITERATOR_NEXT(TransformIterator, ++it, --it)
  ITERATOR_ADVANCE(TransformIterator, i, it += i, it -= i)
  ITERATOR_ARITHMETICP(TransformIterator, a, b, a.it+b)
  ITERATOR_ARITHMETICN(TransformIterator, a, b, a.it-b)
  ITERATOR_COMPARISION(TransformIterator, a, b, a.it, b.it)
};


template <class I, class F>
auto transform(I ib, I ie, F fn) {
  auto b = TransformIterator<I, F>(ib, fn);
  auto e = TransformIterator<I, F>(ie, fn);
  return iterable(b, e);
}

template <class J, class F>
auto transform(const J& x, F fn) {
  auto b = x.begin();
  auto e = x.end();
  return transform(b, e, fn);
}




// FILTER
// ------

template <class I, class F>
class FilterIterator {
  I it;
  const I ie;
  const F fn;

  public:
  ITERATOR_USING_I(I);
  FilterIterator(I ix, I ie, F fn) : it(ix), ie(ie), fn(fn) { while (it!=ie && !fn(*it)) ++it; }
  ITERATOR_DEREF(FilterIterator, *it, it[i], NULL)
  ITERATOR_NEXTP(FilterIterator, do { ++it; } while (it!=ie && !fn(*it)))
  ITERATOR_ADVANCEP(FilterIterator, i, for (; i>0; i--) ++it)
  ITERATOR_ARITHMETICP(FilterIterator, a, b, a.it+b)
  ITERATOR_COMPARISION(FilterIterator, a, b, a.it, b.it)
};


template <class I, class F>
auto filter(I ib, I ie, F fn) {
  auto b = FilterIterator<I, F>(ib, ie, fn);
  auto e = FilterIterator<I, F>(ie, ie, fn);
  return iterable(b, e);
}

template <class J, class F>
auto filter(const J& x, F fn) {
  return filter(x.begin(), x.end(), fn);
}




// RANGE
// -----

template <class T>
int rangeSize(T v, T V, T DV=1) {
  return max(0, (int) ceilDiv(V-v, DV));
}

template <class T>
int rangeLast(T v, T V, T DV=1) {
  return v + DV*(rangeSize(v, V, DV) - 1);
}


template <class T>
class RangeIterator {
  T n;

  public:
  ITERATOR_USING(random_access_iterator_tag, T, T, T, T*)
  RangeIterator(T n) : n(n) {}
  ITERATOR_DEREF(RangeIterator, n, n+i, this)
  ITERATOR_NEXT(RangeIterator, ++n, --n)
  ITERATOR_ADVANCE(RangeIterator, i, n += i, n -= i)
  ITERATOR_ARITHMETICP(RangeIterator, a, b, a.n+b)
  ITERATOR_ARITHMETICN(RangeIterator, a, b, a.n-b)
  ITERATOR_COMPARISION(RangeIterator, a, b, a.n, b.n)
};


template <class T>
auto range(T V) {
  auto b = RangeIterator<T>(0);
  auto e = RangeIterator<T>(V);
  return iterable(b, e);
}

template <class T>
auto range(T v, T V, T DV=1) {
  auto x = range(rangeSize(v, V, DV));
  return transform(x, [=](int n) { return v+DV*n; });
}
