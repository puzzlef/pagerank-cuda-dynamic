#pragma once
#include <cstddef>
#include <iterator>
#include <unordered_map>
#include <algorithm>

using std::ptrdiff_t;
using std::input_iterator_tag;
using std::output_iterator_tag;
using std::forward_iterator_tag;
using std::bidirectional_iterator_tag;
using std::random_access_iterator_tag;
using std::iterator_traits;
using std::unordered_map;
using std::distance;
using std::max;




// ITERATOR-*
// ----------
// Helps create iterators.

#ifndef PAREN_OPEN
#define PAREN_OPEN  (
#define PAREN_CLOSE )
#endif


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

#define ITERATOR_USING_IC(I, cat) \
  using iterator_category = cat; \
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
#define ITERATOR_DEREF(I, n, de, ie) \
  reference operator*() const { return de; } \
  reference operator[](difference_type n) const { return ie; }
#endif


#ifndef ITERATOR_PTR
#define ITERATOR_PTR(I, pe) \
  pointer operator->() { return pe; }
#endif


#ifndef ITERATOR_NEXT
#define ITERATOR_NEXTP(I, ie)  \
  I& operator++() { ie; return *this; }  \
  I operator++(int) { I it = *this; ++(*this); return it; }

#define ITERATOR_NEXTN(I, de) \
  I& operator--() { de; return *this; }  \
  I operator--(int) { I it = *this; --(*this); return it; }

#define ITERATOR_NEXT(I, ie, de) \
  ITERATOR_NEXTP(I, ie) \
  ITERATOR_NEXTN(I, de)
#endif


#ifndef ITERATOR_ADVANCE
#define ITERATOR_ADVANCEP(I, n, fe) \
  I& operator+=(difference_type n) { fe; return *this; }

#define ITERATOR_ADVANCEN(I, n, be) \
  I& operator-=(difference_type n) { be; return *this; }

#define ITERATOR_ADVANCE(I, n, fe, be) \
  ITERATOR_ADVANCEP(I, n, fe) \
  ITERATOR_ADVANCEN(I, n, be)
#endif


#ifndef ITERATOR_ARITHMETICP
#define ITERATOR_ARITHMETICP(I, it, n, ...)  \
  friend I operator+(const I& it, difference_type n) { return __VA_ARGS__; } \
  friend I operator+(difference_type n, const I& it) { return __VA_ARGS__; }
#endif

#ifndef ITERATOR_ARITHMETICN
#define ITERATOR_ARITHMETICN(I, it, n, ...) \
  friend I operator-(const I& it, difference_type n) { return __VA_ARGS__; } \
  friend I operator-(difference_type n, const I& it) { return __VA_ARGS__; }
#endif


#ifndef ITERATOR_COMPARISION
#define ITERATOR_COMPARISION(I, l, r, le, re)  \
  friend bool operator==(const I& l, const I& r) { return le == re; } \
  friend bool operator!=(const I& l, const I& r) { return le != re; } \
  friend bool operator>=(const I& l, const I& r) { return le >= re; } \
  friend bool operator<=(const I& l, const I& r) { return le <= re; } \
  friend bool operator> (const I& l, const I& r) { return le >  re; } \
  friend bool operator< (const I& l, const I& r) { return le <  re; }

#define ITERATOR_COMPARISION_DO3(I, l, r, e0, e1, e2, e3)  \
  friend bool operator==(const I& l, const I& r) { e0 == e1 == e2 == e3; } \
  friend bool operator!=(const I& l, const I& r) { e0 != e1 != e2 != e3; } \
  friend bool operator>=(const I& l, const I& r) { e0 >= e1 >= e2 >= e3; } \
  friend bool operator<=(const I& l, const I& r) { e0 <= e1 <= e2 <= e3; } \
  friend bool operator> (const I& l, const I& r) { e0 >  e1 >  e2 >  e3; } \
  friend bool operator< (const I& l, const I& r) { e0 <  e1 <  e2 <  e3; }
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
  auto begin()  const { return ib; }
  auto end()    const { return ie; }
  size_t size() const { return distance(ib, ie); }
  bool empty()  const { return ib == ie; }
};


template <class I>
auto makeIter(I ib, I ie) {
  return Iterable<I>(ib, ie);
}

template <class J>
auto makeIter(const J& x) {
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
  SizedIterable(I ib, I ie) : Iterable<I>(ib, ie), N(distance(ib, ie)) {}
  size_t size() const { return N; }
  bool empty()  const { return N==0; }
};


template <class I>
auto sizedIter(I ib, I ie, int N) {
  return SizedIterable<I>(ib, ie, N);
}

template <class I>
auto sizedIter(I ib, I ie) {
  return SizedIterable<I>(ib, ie);
}

template <class J>
auto sizedIter(const J& x, int N) {
  using I = decltype(x.begin());
  return SizedIterable<I>(x.begin(), x.end(), N);
}

template <class J>
auto sizedIterable(const J& x) {
  using I = decltype(x.begin());
  return SizedIterable<I>(x.begin(), x.end());
}




// SIZE
// ----

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
auto sliceIter(const J& x, int i) {
  auto b = x.begin(), e = x.end();
  return sizedIter(b+i<e? b+i:e, e);
}

template <class J>
auto sliceIter(const J& x, int i, int I) {
  auto b = x.begin(), e = x.end();
  return sizedIter(b+i<e? b+i:e, b+I<e? b+I:e, I-i);
}




// POINTER
// -------

template <class T>
class PointerIterator {
  using iterator = PointerIterator;
  T *it;

  public:
  ITERATOR_USING(random_access_iterator_tag, ptrdiff_t, T, T&, T*)
  PointerIterator(T *it) : it(it) {}
  ITERATOR_DEREF(iterator, i, *it, it[i])
  ITERATOR_PTR(iterator, &it)
  ITERATOR_NEXT(iterator, ++it, --it)
  ITERATOR_ADVANCE(iterator, i, it += i, it -= i)
  ITERATOR_ARITHMETICP(iterator, a, n, iterator(a.it+n))
  ITERATOR_ARITHMETICN(iterator, a, n, iterator(a.it-n))
  ITERATOR_COMPARISION(iterator, a, b, a.it, b.it)
};

template <class T>
class ConstPointerIterator {
  using iterator = ConstPointerIterator;
  const T* it;

  public:
  ITERATOR_USING(random_access_iterator_tag, ptrdiff_t, T, const T&, const T*)
  ConstPointerIterator(const T* it) : it(it) {}
  ITERATOR_DEREF(iterator, i, *it, it[i])
  ITERATOR_PTR(iterator, &it)
  ITERATOR_NEXT(iterator, ++it, --it)
  ITERATOR_ADVANCE(iterator, i, it += i, it -= i)
  ITERATOR_ARITHMETICP(iterator, a, n, iterator(a.it+n))
  ITERATOR_ARITHMETICN(iterator, a, n, iterator(a.it-n))
  ITERATOR_COMPARISION(iterator, a, b, a.it, b.it)
};


template <class T>
auto pointerIterator(T* it) {
  return PointerIterator<T>(it);
}

template <class T>
auto pointerIterator(const T* it) {
  return ConstPointerIterator<T>(it);
}

template <class T>
auto cpointerIterator(const T* it) {
  return ConstPointerIterator<T>(it);
}


template <class T>
auto pointerIter(T* ib, T* ie) {
  auto b = PointerIterator<T>(ib);
  auto e = PointerIterator<T>(ie);
  return makeIter(b, e);
}

template <class T>
auto pointerIter(const T* ib, const T* ie) {
  auto b = ConstPointerIterator<T>(ib);
  auto e = ConstPointerIterator<T>(ie);
  return makeIter(b, e);
}

template <class T>
auto cpointerIter(const T* ib, const T* ie) {
  auto b = ConstPointerIterator<T>(ib);
  auto e = ConstPointerIterator<T>(ie);
  return makeIter(b, e);
}

template <class J>
auto pointerIter(J& x, int i, int I) {
  return pointerIter(x.data()+i, x.data()+I);
}

template <class J>
auto pointerIter(J& x, int N) {
  return pointerIter(x.data(), x.data()+N);
}

template <class J>
auto pointerIter(J& x) {
  return pointerIter(x.data(), x.data()+x.size());
}

template <class J>
auto pointerIter(const J& x, int i, int I) {
  return pointerIter(x.data()+i, x.data()+I);
}

template <class J>
auto pointerIter(const J& x, int N) {
  return pointerIter(x.data(), x.data()+N);
}

template <class J>
auto pointerIter(const J& x) {
  return pointerIter(x.data(), x.data()+x.size());
}

template <class J>
auto cpointerIter(const J& x, int i, int I) {
  return cpointerIter(x.data()+i, x.data()+I);
}

template <class J>
auto cpointerIter(const J& x, int N) {
  return cpointerIter(x.data(), x.data()+N);
}

template <class J>
auto cpointerIter(const J& x) {
  return cpointerIter(x.data(), x.data()+x.size());
}




// TRANSFORM
// ---------

template <class I, class F>
class TransformIterator {
  using iterator = TransformIterator;
  I it;
  const F fn;

  public:
  ITERATOR_USING_IVR(I, decltype(fn(*it)), value_type)
  TransformIterator(I it, F fn) : it(it), fn(fn) {}
  ITERATOR_DEREF(iterator, i, fn(*it), fn(it[i]))
  ITERATOR_PTR(iterator, NULL)
  ITERATOR_NEXT(iterator, ++it, --it)
  ITERATOR_ADVANCE(iterator, i, it += i, it -= i)
  ITERATOR_ARITHMETICP(iterator, a, b, iterator(a.it+b))
  ITERATOR_ARITHMETICN(iterator, a, b, iterator(a.it-b))
  ITERATOR_COMPARISION(iterator, a, b, a.it, b.it)
};


template <class I, class F>
auto transformIter(I ib, I ie, F fn) {
  auto b = TransformIterator<I, F>(ib, fn);
  auto e = TransformIterator<I, F>(ie, fn);
  return makeIter(b, e);
}

template <class J, class F>
auto transformIter(const J& x, F fn) {
  return transformIter(x.begin(), x.end(), fn);
}




// FILTER
// ------

template <class I, class F>
class FilterIterator {
  using iterator = FilterIterator;
  I it;
  const I ie;
  const F fn;

  public:
  ITERATOR_USING_I(I);
  FilterIterator(I ix, I ie, F fn) : it(ix), ie(ie), fn(fn) { while (it!=ie && !fn(*it)) ++it; }
  ITERATOR_DEREF(iterator, i, *it, it[i])
  ITERATOR_PTR(iterator, it.I::operator->())
  ITERATOR_NEXTP(iterator, do { ++it; } while (it!=ie && !fn(*it)))
  ITERATOR_ADVANCEP(iterator, i, for (; i>0; i--) ++it)
  ITERATOR_ARITHMETICP(iterator, a, b, iterator(a.it+b))
  ITERATOR_COMPARISION(iterator, a, b, a.it, b.it)
};


template <class I, class F>
auto filterIter(I ib, I ie, F fn) {
  auto b = FilterIterator<I, F>(ib, ie, fn);
  auto e = FilterIterator<I, F>(ie, ie, fn);
  return makeIter(b, e);
}

template <class J, class F>
auto filterIter(const J& x, F fn) {
  return filterIter(x.begin(), x.end(), fn);
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
  using iterator = RangeIterator;
  T n;

  public:
  ITERATOR_USING(random_access_iterator_tag, T, T, T, const T*)
  RangeIterator(T n) : n(n) {}
  ITERATOR_DEREF(iterator, i, n, n+i)
  ITERATOR_PTR(iterator, &n)
  ITERATOR_NEXT(iterator, ++n, --n)
  ITERATOR_ADVANCE(iterator, i, n += i, n -= i)
  ITERATOR_ARITHMETICP(iterator, a, b, iterator(a.n+b))
  ITERATOR_ARITHMETICN(iterator, a, b, iterator(a.n-b))
  ITERATOR_COMPARISION(iterator, a, b, a.n, b.n)
};


template <class T>
auto rangeIter(T V) {
  auto b = RangeIterator<T>(0);
  auto e = RangeIterator<T>(V);
  return makeIter(b, e);
}

template <class T>
auto rangeIter(T v, T V, T DV=1) {
  auto x = rangeIter(rangeSize(v, V, DV));
  return transformIter(x, [=](int n) { return v+DV*n; });
}




// DEFAULT
// -------
// Return default value of type, always.

template <class T>
class DefaultIterator {
  using iterator = DefaultIterator;
  const T x;

  public:
  ITERATOR_USING(random_access_iterator_tag, ptrdiff_t, T, const T&, const T*)
  DefaultIterator() : x() {}
  ITERATOR_DEREF(iterator, i, x, x)
  ITERATOR_PTR(iterator, &x)
  ITERATOR_NEXT(iterator, {}, {})
  ITERATOR_ADVANCE(iterator, i, {}, {})
  ITERATOR_ARITHMETICP(iterator, it, n, it)
  ITERATOR_ARITHMETICN(iterator, it, n, it)
  ITERATOR_COMPARISION(iterator, l, r, 0, 0)
};


template <class T>
auto defaultIterator(const T& _) {
  return DefaultIterator<T>();
}


template <class T>
class DefaultValueIterator {
  using iterator = DefaultValueIterator;

  public:
  ITERATOR_USING(random_access_iterator_tag, ptrdiff_t, T, T, const T*)
  DefaultValueIterator() {}
  ITERATOR_DEREF(iterator, i, T(), T())
  ITERATOR_PTR(iterator, NULL)
  ITERATOR_NEXT(iterator, {}, {})
  ITERATOR_ADVANCE(iterator, i, {}, {})
  ITERATOR_ARITHMETICP(iterator, it, n, it)
  ITERATOR_ARITHMETICN(iterator, it, n, it)
  ITERATOR_COMPARISION(iterator, l, r, 0, 0)
};

template <class T>
auto defaultValueIterator(const T& _) {
  return DefaultValueIterator<T>();
}




// TERNARY
// -------
// Select iterator by boolean.

template <class I1, class I0>
class TernaryIterator {
  using iterator = TernaryIterator;
  using T = typename I1::value_type;
  const bool sel;
  I1 i1;
  I0 i0;

  public:
  ITERATOR_USING_IC(I1, random_access_iterator_tag)
  TernaryIterator(bool sel, I1 i1, I0 i0) : sel(sel), i1(i1), i0(i0) {}
  ITERATOR_DEREF(iterator, n, sel? *i1 : *i0, sel? i1[n] : i0[n])
  ITERATOR_PTR(iterator, sel? i1.I1::operator->() : i0.I0::operator->())
  ITERATOR_NEXTP(iterator, if (sel) { ++i1; } else { ++i0; })
  ITERATOR_NEXTN(iterator, if (sel) { --i1; } else { --i0; })
  ITERATOR_ADVANCEP(iterator, n, if (sel) { i1+=n; } else { i0+=n; })
  ITERATOR_ADVANCEN(iterator, n, if (sel) { i1-=n; } else { i0-=n; })
  ITERATOR_ARITHMETICP(iterator, it, n, it.sel? iterator(it.i0, it.i1+n) : iterator(it.i0+n, it.i1))
  ITERATOR_ARITHMETICN(iterator, it, n, it.sel? iterator(it.i0, it.i1-n) : iterator(it.i0-n, it.i1))
  ITERATOR_COMPARISION_DO3(iterator, l, r, if (l.sel!=r.sel) return ! PAREN_OPEN 0, 0 PAREN_CLOSE; return l.sel? l.i1, r.i1 : l.i0, r.i0)
};


template <class I1, class I0>
auto ternaryIterator(bool sel, I1 i1, I0 i0) {
  return TernaryIterator<I1, I0>(sel, i1, i0);
}

template <class J1, class J0>
auto ternaryIter(bool sel, const J1& x1, const J0& x0) {
  auto b = ternaryIterator(sel, x1.begin(), x0.begin());
  auto e = ternaryIterator(sel, x1.end(), x0.end());
  return makeIter(b, e);
}




// SELECT
// ------
// Select iterator by index.
// Can be done using tuples.
