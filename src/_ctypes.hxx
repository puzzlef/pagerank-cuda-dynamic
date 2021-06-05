#pragma once




// NONE
// ----
// Zero size type.

#ifndef NONE
struct None {
  friend bool operator==(None a, None b) noexcept { return true; }

  template <class T>
  friend bool operator==(None a, const T& b) noexcept { return false; }

  template <class T>
  friend bool operator==(const T& a, None b) noexcept { return false; }
};
#define NONE None
#endif
