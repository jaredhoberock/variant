#include <iostream>
#include "variant.hpp"
#include <cassert>
#include <string>

int main()
{
  variant<char, int, float, double> v1;
  assert(v1.index() == 0);
  assert(get<0>(v1) == 0);

  variant<char, int, float, double> v2 = 13;
  assert(v2.index() == 1);
  assert(get<1>(v2) == 13);

  variant<char, int, float, double> v3 = 13;
  assert(v3.index() == 1);
  assert(get<1>(v3) == 13);

  assert(v2 == v3);
  assert(v1 != v3);

  std::cout << "v1: " << v1 << std::endl;
  std::cout << "v2: " << v2 << std::endl;
  std::cout << "v3: " << v3 << std::endl;

  variant<char, int, float, double, std::string> v4 = std::string("hello, world!");
  assert(v4.index() == 4);
  assert(get<4>(v4) == "hello, world!");

  std::cout << "v4: " << v4 << std::endl;

  auto s = get<4>(std::move(v4));
  assert(s == "hello, world!");
  assert(get<4>(v4) == "");

  variant<float, double> v5 = 7.f;
  variant<float, double> v6 = 13.;
  assert(v5.index() == 0);
  assert(get<0>(v5) == 7.f);
  assert(v6.index() == 1);
  assert(get<1>(v6) == 13.);

  v5.swap(v6);

  assert(v5.index() == 1);
  std::cout << "v5: " << v5 << std::endl;
  assert(get<1>(v5) == 13.);
  assert(v6.index() == 0);
  assert(get<0>(v6) == 7.f);

  return 0;
}

