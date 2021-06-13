#pragma once
#include <cmath>
#include <type_traits>

using std::is_floating_point;
using std::ceil;




// CEIL-DIV
// --------

template <class T>
T ceilDiv(T x, T y) {
  if (is_floating_point<T>()) return ceil(x/y);
  else return (x + y-1) / y;
}




// SGN
// ---
// https://stackoverflow.com/a/4609795/1413259

template <typename T>
int sgn(T x) {
  return (T() < x) - (x < T());
}
