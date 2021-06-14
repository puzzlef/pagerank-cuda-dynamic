#pragma once
#include "_main.hxx"




// ADJUST-RANKS
// ------------

template <class T, class J>
void adjustRanks(vector<T>& a, const vector<T>& r, J&& ks0, J&& ks1, T radd, T rmul, T rset) {
  auto ksNew = setDifference(ks1, ks0);
  for (int k : ks0)
    a[k] = (r[k]+radd)*rmul;
  for (int k : ksNew)
    a[k] = rset;
}

template <class T, class J>
auto adjustRanks(int N, const vector<T>& r, J&& ks0, J&& ks1, T radd, T rmul, T rset) {
  vector<T> a(N); adjustRanks(a, r, ks0, ks1, radd, rmul, rset);
  return a;
}
