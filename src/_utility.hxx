#pragma once
#include <chrono>

using std::chrono::microseconds;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;




// MEASURE
// -------

template <class F>
float measureDuration(F fn, int N=1) {
  auto start = high_resolution_clock::now();
  for (int i=0; i<N; i++)
    fn();
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  return duration.count()/(N*1000.0f);
}


template <class F>
float measureDurationMarked(F fn, int N=1) {
  float duration = 0;
  for (int i=0; i<N; i++)
    fn([&](auto fm) { duration += measureDuration(fm); });
  return duration/N;
}




// RETRY
// -----

template <class F>
bool retry(F fn, int N=2) {
  for (int i=0; i<N; i++)
    if (fn()) return true;
  return false;
}
