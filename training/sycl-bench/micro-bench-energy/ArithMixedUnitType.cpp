#include "common.h"
#include <sycl/sycl.hpp>

template <size_t AddPercFloat, size_t MulPercFloat, size_t DivPercFloat, size_t AddPercInt, size_t MulPercInt,
    size_t DivPercInt, size_t SpecPerc>
class ArithMixedUnitType {
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
  ArithMixedUnitType(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size;
    num_iters = args.num_iterations;

    in_array.resize(size);
    out_array1.resize(size);
    out_array2.resize(size);
    out_array3.resize(size);
    out_array4.resize(size);

    for(size_t i = 0; i < size; i++) {
      in_array[i] = (float)(i % 1) + 1;
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
        float raf, rbf, rcf, rdf;
        int rai, rbi, rci, rdi;

        if(gid < size) {
          rai = raf = in_acc[gid];
          rbi = rbf = in_acc[size - gid];
          rci = rcf = in_acc[gid];
          rdi = rdf = in_acc[size - gid];

          for(size_t i = 0; i < num_iters; i++) {

#pragma unroll
            for(int i = 0; i < AddPercFloat; i++) {
              raf = raf + rbf;
              rbf = rbf + rcf;
              rcf = rcf + rdf;
              rdf = rdf + raf;
            }
#pragma unroll
            for(int i = 0; i < DivPercFloat; i++) {
              raf = raf / rbf;
              rbf = rbf / rcf;
              rcf = rcf / rdf;
              rdf = rdf / raf;
            }
#pragma unroll
            for(int i = 0; i < MulPercFloat; i++) {
              raf = raf * rbf;
              rbf = rbf * rcf;
              rcf = rcf * rdf;
              rdf = rdf * raf;
            }

#pragma unroll
            for(int i = 0; i < AddPercInt; i++) {
              rai = rai + rbi;
              rbi = rbi + rci;
              rci = rci + rdi;
              rdi = rdi + rai;
            }
#pragma unroll
            for(int i = 0; i < DivPercInt; i++) {
              rai = rai / rbi;
              rbi = rbi / rci;
              rci = rci / rdi;
              rdi = rdi / rai;
            }
#pragma unroll
            for(int i = 0; i < MulPercInt; i++) {
              rai = rai * rbi;
              rbi = rbi * rci;
              rci = rci * rdi;
              rdi = rdi * rai;
            }
#pragma unroll
            for(int i = 0; i < SpecPerc; i++) {
              raf = log(rbf);
              rbf = cos(rcf);
              rcf = log(rdf);
              rdf = sin(raf);
            }
          }

          out_acc1[gid] = raf + rai;
          out_acc2[gid] = rbf + rbi;
          out_acc3[gid] = rcf + rci;
          out_acc4[gid] = rdf + rdi;
        }
      });
    }));
  }

  static std::string getBenchmarkName() {
    std::string name = "ArithMixedUnitType_";
    name.append(std::to_string(AddPercFloat))
        .append("_")
        .append(std::to_string(MulPercFloat))
        .append("_")
        .append(std::to_string(DivPercFloat))
        .append("_")
        .append(std::to_string(AddPercInt))
        .append("_")
        .append(std::to_string(MulPercInt))
        .append("_")
        .append(std::to_string(DivPercInt))
        .append("_")
        .append(std::to_string(SpecPerc));
    return name;
  }

  bool verify(VerificationSetting& ver) { return true; }
};

// 1000000 25000
int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<ArithMixedUnitType<1, 1, 1, 1, 1, 1, 0>>();
  app.run<ArithMixedUnitType<3, 3, 3, 3, 3, 3, 0>>();
  app.run<ArithMixedUnitType<5, 5, 5, 5, 5, 5, 0>>();

  app.run<ArithMixedUnitType<3, 3, 3, 1, 1, 1, 0>>();
  app.run<ArithMixedUnitType<1, 1, 1, 3, 3, 3, 0>>();
  app.run<ArithMixedUnitType<5, 5, 5, 2, 2, 2, 0>>();
  app.run<ArithMixedUnitType<2, 2, 2, 5, 5, 5, 0>>();

  app.run<ArithMixedUnitType<1, 1, 1, 1, 1, 1, 1>>();
  app.run<ArithMixedUnitType<3, 3, 3, 3, 3, 3, 1>>();
  app.run<ArithMixedUnitType<5, 5, 5, 5, 5, 5, 1>>();

  app.run<ArithMixedUnitType<3, 3, 3, 1, 1, 1, 1>>();
  app.run<ArithMixedUnitType<1, 1, 1, 3, 3, 3, 1>>();
  app.run<ArithMixedUnitType<5, 5, 5, 2, 2, 2, 1>>();
  app.run<ArithMixedUnitType<2, 2, 2, 5, 5, 5, 1>>();

  return 0;
}