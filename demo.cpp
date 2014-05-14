#include <iostream>
#include "variant.hpp"
#include <typeinfo>
#include <cassert>
#include <string>

int main()
{
  variant<char, int, float, double> v1;
  assert(typeid(char) == v1.type());

  std::cout << "expect char: " << v1.type().name() << std::endl;

  variant<char, int, float, double> v2 = 13;
  assert(typeid(int) == v2.type());

  std::cout << "expect int: " << v2.type().name() << std::endl;

  variant<char, int, float, double> v3 = 13;

  std::cout << "expect true: "  << (v2 == v3) << std::endl;
  std::cout << "expect false: " << (v2 != v3) << std::endl;
  assert(v2 == v3);

  std::cout << "expect true: "  << (v1 != v3) << std::endl;
  std::cout << "expect false: " << (v1 == v3) << std::endl;
  assert(v1 != v3);

  std::cout << "v1: " << v1 << std::endl;
  std::cout << "v2: " << v2 << std::endl;
  std::cout << "v3: " << v3 << std::endl;

  variant<char, int, float, double, std::string> v4 = "hello, world!";

  std::cout << "v4: " << v4 << std::endl;

  return 0;
}

