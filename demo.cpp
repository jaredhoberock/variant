#include <iostream>
#include "variant.hpp"
#include <cassert>
#include <string>

int main()
{
  using namespace std::experimental;

  variant<char, int, float, double> v1;
  assert(v1.index() == 0);
  assert(get<0>(v1) == 0);
  assert(get<char>(v1) == 0);
  assert(holds_alternative<char>(v1));

  variant<char, int, float, double> v2 = 13;
  assert(v2.index() == 1);
  assert(get<1>(v2) == 13);
  assert(get<int>(v2) == 13);
  assert(holds_alternative<int>(v2));

  variant<char, int, float, double> v3 = 13;
  assert(v3.index() == 1);
  assert(get<1>(v3) == 13);
  assert(get<int>(v3) == 13);
  assert(holds_alternative<int>(v3));

  assert(v2 == v3);
  assert(v1 != v3);

  std::cout << "v1: " << v1 << std::endl;
  std::cout << "v2: " << v2 << std::endl;
  std::cout << "v3: " << v3 << std::endl;

  variant<char, int, float, double, std::string> v4 = std::string("hello, world!");
  assert(v4.index() == 4);
  assert(get<4>(v4) == "hello, world!");
  assert(get<std::string>(v4) == "hello, world!");
  assert(holds_alternative<std::string>(v4));

  std::cout << "v4: " << v4 << std::endl;

  auto s = get<4>(std::move(v4));
  assert(s == "hello, world!");
  assert(get<4>(v4) == "");
  assert(get<std::string>(v4) == "");
  assert(holds_alternative<std::string>(v4));

  variant<float, double> v5 = 7.f;
  variant<float, double> v6 = 13.;
  assert(v5.index() == 0);
  assert(get<0>(v5) == 7.f);
  assert(get<float>(v5) == 7.f);
  assert(holds_alternative<float>(v5));
  assert(v6.index() == 1);
  assert(get<1>(v6) == 13.);
  assert(get<double>(v6) == 13.);
  assert(holds_alternative<double>(v6));

  v5.swap(v6);

  assert(v5.index() == 1);
  assert(get<1>(v5) == 13.);
  assert(get<double>(v5) == 13.);
  assert(holds_alternative<double>(v5));

  assert(v6.index() == 0);
  assert(get<0>(v6) == 7.f);
  assert(get<float>(v6) == 7.f);
  assert(holds_alternative<float>(v6));

  variant<char, int, float, double> v7 = 13.;
  try
  {
    auto got = get<0>(v7);
    assert(0);
  }
  catch(bad_variant_access&)
  {
  }

  std::cout << "OK" << std::endl;

  return 0;
}

