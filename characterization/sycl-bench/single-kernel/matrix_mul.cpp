#include <iostream>
#include <sycl/sycl.hpp>

#include "common.h"


namespace s = sycl;
class MatrixMulAccKernel; // kernel forward declaration

template <class T>
class matrixMul {
private:
  int size;
  int num_iters;
  const s::accessor<T, 1, s::access_mode::read> in_A;
  const s::accessor<T, 1, s::access_mode::read> in_B;
  s::accessor<T, 1, s::access_mode::read_write> out;

public:
  matrixMul(int size, int num_iters, const s::accessor<T, 1, s::access_mode::read> in_A,
      const s::accessor<T, 1, s::access_mode::read> in_B, s::accessor<T, 1, s::access_mode::read_write> out)
      : size(size), num_iters(num_iters), in_A(in_A), in_B(in_B), out(out) {}

  void operator()(s::id<2> gid) const {
    int gidx = gid.get(0);
    int gidy = gid.get(1);
    for(int iter = 0; iter < num_iters; iter++)
      for(int k = 0; k < size; k++) out[gidx * size + gidy] += in_A[gidx * size + k] * in_B[k * size + gidy];
  }
};

template <class T>
class MatrixMulAcc {
protected:
  size_t num_iters;
  std::vector<T> a;
  std::vector<T> b;
  std::vector<T> c;

  PrefetchedBuffer<T, 1> a_buf;
  PrefetchedBuffer<T, 1> b_buf;
  PrefetchedBuffer<T, 1> c_buf;

  size_t size;
  BenchmarkArgs& args;

public:
  MatrixMulAcc(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size; // input size defined by the user
    num_iters = args.num_iterations;
    a.resize(size * size);
    b.resize(size * size);
    c.resize(size * size);

    std::fill(a.begin(), a.end(), 1);
    std::fill(b.begin(), b.end(), 1);
    std::fill(c.begin(), c.end(), 0);

    a_buf.initialize(args.device_queue, a.data(), s::range<1>{size * size});
    b_buf.initialize(args.device_queue, b.data(), s::range<1>{size * size});
    c_buf.initialize(args.device_queue, c.data(), s::range<1>{size * size});
  }

  void run(std::vector<sycl::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
      auto acc_a = a_buf.template get_access<s::access_mode::read>(cgh);
      auto acc_b = b_buf.template get_access<s::access_mode::read>(cgh);
      auto acc_c = c_buf.template get_access<s::access_mode::read_write>(cgh);
      cgh.parallel_for(s::range<2>{size, size}, matrixMul<T>(size, num_iters, acc_a, acc_b, acc_c)); // end parallel for
    })); // end events.push back
  }


  bool verify(VerificationSetting& ver) {
    c_buf.reset();
    for(int i = 0; i < size * size; i++)
      if(size != c[i])
        return false;


    return true;
  }

  static std::string getBenchmarkName() { return "Matrix_mul"; }
};


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<MatrixMulAcc<float>>();
  return 0;
}
