#include "common.h"
#include <sycl/sycl.hpp>

template <typename ValueType, size_t AddPerc, size_t MulPerc, size_t DivPerc, size_t SpecPerc>
class ArithSingleUnit {
protected:
  std::vector<float> in_array;
  std::vector<float> out_array1;
  std::vector<float> out_array2;
  std::vector<float> out_array3;
  std::vector<float> out_array4;

  PrefetchedBuffer<float, 1> in_buf;
  PrefetchedBuffer<float, 1> out_buf1;
  PrefetchedBuffer<float, 1> out_buf2;
  PrefetchedBuffer<float, 1> out_buf3;
  PrefetchedBuffer<float, 1> out_buf4;

  size_t size;
  size_t num_iters;

  BenchmarkArgs& args;

public:
  ArithSingleUnit(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size;
    num_iters = args.num_iterations;

    in_array.resize(size);
    out_array1.resize(size);
    out_array2.resize(size);
    out_array3.resize(size);
    out_array4.resize(size);

    for(size_t i = 0; i < size; i++) {
      in_array[i] = (ValueType)(i % 1) + 1;
    }

    // buffer initialization
    in_buf.initialize(args.device_queue, in_array.data(), sycl::range<1>{size});
    out_buf1.initialize(args.device_queue, out_array1.data(), sycl::range<1>{size});
    out_buf2.initialize(args.device_queue, out_array2.data(), sycl::range<1>{size});
    out_buf3.initialize(args.device_queue, out_array3.data(), sycl::range<1>{size});
    out_buf4.initialize(args.device_queue, out_array4.data(), sycl::range<1>{size});
  }

  void run(std::vector<sycl::event>& events) {
    events.push_back(args.device_queue.submit([&](sycl::handler& cgh) {
      auto in_acc = in_buf.get_access<sycl::access_mode::read>(cgh);
      auto out_acc1 = out_buf1.get_access<sycl::access_mode::write>(cgh);
      auto out_acc2 = out_buf2.get_access<sycl::access_mode::write>(cgh);
      auto out_acc3 = out_buf3.get_access<sycl::access_mode::write>(cgh);
      auto out_acc4 = out_buf4.get_access<sycl::access_mode::write>(cgh);

      sycl::range<1> grid{size};

      cgh.parallel_for(grid, [=, num_iters = num_iters, size = size](sycl::id<1> id) {
        int gid = id.get(0);
        float ra, rb, rc, rd;

        if(gid < size) {
          ra = in_acc[gid];
          rb = in_acc[size - gid];
          rc = in_acc[gid];
          rd = in_acc[size - gid];

          for(size_t i = 0; i < num_iters; i++) {

#pragma unroll
            for(int i = 0; i < AddPerc; i++) {
              ra = ra + rb;
              rb = rb + rc;
              rc = rc + rd;
              rd = rd + ra;
            }
#pragma unroll
            for(int i = 0; i < DivPerc; i++) {
              ra = ra / rb;
              rb = rb / rc;
              rc = rc / rd;
              rd = rd / ra;
            }
#pragma unroll
            for(int i = 0; i < MulPerc; i++) {
              ra = ra * rb;
              rb = rb * rc;
              rc = rc * rd;
              rd = rd * ra;
            }
#pragma unroll
            for(int i = 0; i < SpecPerc; i++) {
              ra = log(rb);
              rb = cos(rc);
              rc = log(rd);
              rd = sin(ra);
            }
          }

          out_acc1[gid] = ra;
          out_acc2[gid] = rb;
          out_acc3[gid] = rc;
          out_acc4[gid] = rd;
        }
      });
    }));
  }

  static std::string getBenchmarkName() {
    std::string name = "ArithSingleUnit_";
    name.append(std::is_same_v<ValueType, int> ? "int" : "float")
        .append("_")
        .append(std::to_string(AddPerc))
        .append("_")
        .append(std::to_string(MulPerc))
        .append("_")
        .append(std::to_string(DivPerc))
        .append("_")
        .append(std::to_string(SpecPerc));
    return name;
  }

  bool verify(VerificationSetting& ver) { return true; }
};

// 1000000 500000
int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  // std::cout << "int\n";
  // std::cout << "add\n";
  app.run<ArithSingleUnit<int, 1, 0, 0, 0>>();
  app.run<ArithSingleUnit<int, 5, 0, 0, 0>>();
  app.run<ArithSingleUnit<int, 10, 0, 0, 0>>();

  // std::cout << "mul\n";
  app.run<ArithSingleUnit<int, 0, 1, 0, 0>>();
  app.run<ArithSingleUnit<int, 0, 5, 0, 0>>();
  app.run<ArithSingleUnit<int, 0, 10, 0, 0>>();

  // std::cout << "div\n";
  app.run<ArithSingleUnit<int, 0, 0, 1, 0>>();
  app.run<ArithSingleUnit<int, 0, 0, 2, 0>>();
  app.run<ArithSingleUnit<int, 0, 0, 3, 0>>();

  // std::cout << "\nfloat\n";
  // std::cout << "add\n";
  app.run<ArithSingleUnit<float, 1, 0, 0, 0>>();
  app.run<ArithSingleUnit<float, 5, 0, 0, 0>>();
  app.run<ArithSingleUnit<float, 10, 0, 0, 0>>();

  // std::cout << "mul\n";
  app.run<ArithSingleUnit<float, 0, 1, 0, 0>>();
  app.run<ArithSingleUnit<float, 0, 5, 0, 0>>();
  app.run<ArithSingleUnit<float, 0, 10, 0, 0>>();

  // std::cout << "div\n";
  app.run<ArithSingleUnit<float, 0, 0, 1, 0>>();
  app.run<ArithSingleUnit<float, 0, 0, 2, 0>>();
  app.run<ArithSingleUnit<float, 0, 0, 3, 0>>();

  // std::cout << "\nspecial\n";
  app.run<ArithSingleUnit<float, 0, 0, 0, 1>>();
  app.run<ArithSingleUnit<float, 0, 0, 0, 2>>();
  app.run<ArithSingleUnit<float, 0, 0, 0, 3>>();

  return 0;
}