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
