#pragma once
#include <string>

using std::string;




int countLines(const char *x) {
  int a = 1;
  for (; *x; x++) {
    if (*x == '\r' || *x == '\n') a++;
    else if (*x == '\r' && *(x+1) == '\n') x++;
  }
  return a;
}

int countLines(const string& x) {
  return countLines(x.c_str());
}
