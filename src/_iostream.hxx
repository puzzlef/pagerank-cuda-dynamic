#pragma once
#include <string>
#include <vector>
#include <ostream>
#include <fstream>
#include <iostream>

using std::ios;
using std::string;
using std::vector;
using std::ostream;
using std::ifstream;
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

template <class T>
void write(ostream& a, const T *x, int N) {
  a << "{";
  for (int i=0; i<N; i++)
    a << " " << x[i];
  a << " }";
}

template <class T>
void write(ostream& a, const vector<T>& x) {
  write(a, x.data(), x.size());
}

template <class T>
void write(ostream& a, const vector<vector<T>>& x) {
  a << "{\n";
  for (const auto& v : x) {
    a << "  "; write(a, v);
  }
  a << "}";
}




// PRINT
// -----

template <class T>
void print(const T *x, int N) { write(cout, x, N); }

template <class T>
void print(const vector<T>& x) { write(cout, x); }

template <class T>
void print(const vector<vector<T>>& x) { write(cout, x); }




// PRINTLN
// -------

template <class T>
void println(const T *x, int N) { print(x, N); cout << "\n"; }

template <class T>
void println(const vector<T>& x) { print(x); cout << "\n"; }

template <class T>
void println(const vector<vector<T>>& x) { print(x); cout << "\n"; }
