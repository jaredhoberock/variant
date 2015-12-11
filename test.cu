#include <iostream>
#include "variant.hpp"
#include <typeinfo>
#include <cassert>
#include <string>

__host__ __device__
void test()
{
  using namespace std::experimental;

  variant<char, int, float, double> v1;
  assert(v1.index() == 0);

  variant<char, int, float, double> v2 = 13;
  assert(v2.index() == 1);

  variant<char, int, float, double> v3 = 13;
  assert(v3.index() == 1);

  assert(v2 == v3);
  assert(!(v2 != v3));

  assert(v1 != v3);
  assert(!(v1 == v3));
}

__global__ void kernel()
{
  test();
}

int main()
{
  test();

  kernel<<<1,1>>>();

  assert(cudaDeviceSynchronize() == cudaSuccess);

  std::cout << "OK" << std::endl;

  return 0;
}

