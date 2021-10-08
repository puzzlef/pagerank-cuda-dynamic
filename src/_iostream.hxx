#pragma once
#include <utility>
#include <string>
#include <vector>
#include <ostream>
#include <fstream>
#include <iostream>
#include <type_traits>

using std::pair;
using std::string;
using std::vector;
using std::ios;
using std::ostream;
using std::ifstream;
using std::is_fundamental;
using std::cout;




// READ-FILE
// ---------

string readFile(const char *pth) {
  string a; ifstream f(pth);
  f.seekg(0, ios::end);
  a.resize(f.tellg());
  f.seekg(0);
  f.read((char*) a.data(), a.size());
  return a;
}




// WRITE
// -----

template <class K, class V>
ostream& operator<<(ostream& a, const pair<K, V>& x) {
  a << x.first << ": " << x.second;
  return a;
}

template <class T>
ostream& operator<<(ostream& a, const vector<T>& x) {
  if (is_fundamental<T>::value) {
    a << "{";
    for (T v : x)
      a << " " << v;
    a << " }";
  }
  else {
    a << "{\n";
    for (const T& v : x)
      a << "  " << v << "\n";
    a << "}";
  }
  return a;
}




// PRINT*
// ------

template <class T>
void print(const T& x) {cout << x; }

template <class T>
void println(const T& x) { cout << x << "\n"; }
