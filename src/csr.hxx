#pragma once
#include <numeric>
#include <algorithm>
#include <vector>
#include "_main.hxx"

using std::vector;
using std::iota;
using std::equal;
using std::transform;




// SOURCE OFFSETS
// --------------

template <class G, class KS>
inline auto sourceOffsets(const G& x, const KS& ks) {
  vector<size_t> a; size_t i = 0;
  a.reserve(x.order()+1);
  for (auto u : ks) {
    a.push_back(i);
    i += x.degree(u);
  }
  a.push_back(i);
  return a;
}
template <class G>
inline auto sourceOffsets(const G& x) {
  return sourceOffsets(x, x.vertexKeys());
}




// DESTINATION INDICES
// -------------------

template <class G, class KS>
inline auto destinationIndices(const G& x, const KS& ks) {
  using  K = typename G::key_type;
  auto ids = valueIndicesUnorderedMap(ks); vector<K> a;
  for (auto u : ks)
    x.forEachEdgeKey(u, [&](auto v) { a.push_back(K(ids[v])); });
  return a;
}
template <class G>
inline auto destinationIndices(const G& x) {
  return destinationIndices(x, x.vertexKeys());
}
