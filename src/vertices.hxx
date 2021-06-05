#pragma once
#include <vector>
#include "_main.hxx"

using std::vector;




// VERTICES
// --------

template <class G, class F>
auto vertices(const G& x, F fn) {
  using K = decltype(fn(0));
  vector<K> a;
  a.reserve(x.order());
  for (int u : x.vertices())
    a.push_back(fn(u));
  return a;
}

template <class G>
auto vertices(const G& x) {
  return vertices(x, [](int u) { return u; });
}




// VERTEX-DATA
// -----------

template <class G, class J, class F>
auto vertexData(const G& x, J&& ks, F fn) {
  using V = decltype(fn(0));
  vector<V> a;
  a.reserve(x.order());
  for (int u : ks)
    a.push_back(fn(u));
  return a;
}

template <class G, class J>
auto vertexData(const G& x, J&& ks) {
  return vertexData(x, ks, [&](int u) { return x.vertexData(u); });
}

template <class G>
auto vertexData(const G& x) {
  return vertexData(x, x.vertices());
}




// CONTAINER
// ---------

template <class G, class T>
auto createContainer(const G& x, const T& _) {
  return vector<T>(x.span());
}

template <class G, class T>
auto createCompressedContainer(const G& x, const T& _) {
  return vector<T>(x.order());
}


template <class G, class T, class J>
void decompressContainer(vector<T>& a, const G& x, const vector<T>& vs, J&& ks) {
  scatter(a, vs, ks);
}

template <class G, class T>
void decompressContainer(vector<T>& a, const G& x, const vector<T>& vs) {
  decompressContainer(a, x, vs, x.vertices());
}

template <class G, class T, class J>
auto decompressContainer(const G& x, const vector<T>& vs, J&& ks) {
  auto a = createContainer(x, T());
  decompressContainer(a, x, vs, ks);
  return a;
}

template <class G, class T>
auto decompressContainer(const G& x, const vector<T>& vs) {
  return decompressContainer(x, vs, x.vertices());
}


template <class G, class T, class J>
void compressContainer(vector<T>& a, const G& x, const vector<T>& vs, J&& ks) {
  gather(a, vs, ks);
}

template <class G, class T>
void compressContainer(vector<T>& a, const G& x, const vector<T>& vs) {
  return compressContainer(a, x, vs, x.vertices());
}

template <class G, class T, class J>
auto compressContainer(const G& x, const vector<T>& vs, J&& ks) {
  auto a = createCompressedContainer(x, T());
  compressContainer(a, x, vs, ks);
  return a;
}

template <class G, class T>
auto compressContainer(const G& x, const vector<T>& vs) {
  return compressContainer(x, vs, x.vertices());
}
