#include "common.h"
#include <sycl/sycl.hpp>


template <size_t LocalSize, size_t AddPercFloat, size_t MulPercFloat, size_t DivPercFloat, size_t AddPercInt,
    size_t MulPercInt, size_t DivPercInt, size_t SpecPerc>
class ArithLocalMixed {
protected:
  size_t global_size;
  size_t local_size = LocalSize;
  size_t num_iters;

  BenchmarkArgs& args;

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


public:
  ArithLocalMixed(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    global_size = args.problem_size;
    num_iters = args.num_iterations;

    in_array.resize(global_size);
    out_array1.resize(global_size);
    out_array2.resize(global_size);
    out_array3.resize(global_size);
    out_array4.resize(global_size);

    for(size_t i = 0; i < global_size; i++) {
      in_array[i] = (float)(i % 1) + 1;
    }

    in_buf.initialize(args.device_queue, in_array.data(), sycl::range<1>{global_size});
    out_buf1.initialize(args.device_queue, out_array1.data(), sycl::range<1>{global_size});
    out_buf2.initialize(args.device_queue, out_array2.data(), sycl::range<1>{global_size});
    out_buf3.initialize(args.device_queue, out_array3.data(), sycl::range<1>{global_size});
    out_buf4.initialize(args.device_queue, out_array4.data(), sycl::range<1>{global_size});
  }

  void run(std::vector<sycl::event>& events) {
    events.push_back(args.device_queue.submit([&](sycl::handler& cgh) {
      auto in_acc = in_buf.get_access<sycl::access_mode::read>(cgh);
      auto out_acc1 = out_buf1.get_access<sycl::access_mode::write>(cgh);
      auto out_acc2 = out_buf2.get_access<sycl::access_mode::write>(cgh);
      auto out_acc3 = out_buf3.get_access<sycl::access_mode::write>(cgh);
      auto out_acc4 = out_buf4.get_access<sycl::access_mode::write>(cgh);

      sycl::local_accessor<float, 1> in_float_local_acc{sycl::range<1>{LocalSize}, cgh};
      sycl::local_accessor<int, 1> in_int_local_acc{sycl::range<1>{LocalSize}, cgh};

      sycl::range<1> r{global_size};
      sycl::range<1> local_r{LocalSize};

      cgh.parallel_for(sycl::nd_range<1>{r, local_r}, [=, num_iters = num_iters](sycl::nd_item<1> it) {
        sycl::group<1> group = it.get_group();
        int gid = it.get_global_id(0);
        int lid = it.get_local_id(0);

        in_float_local_acc[lid] = in_acc[gid];
        in_int_local_acc[lid] = in_acc[gid];

        sycl::group_barrier(group);

        for(size_t i = 0; i < num_iters; i++) {

#pragma unroll
          for(size_t j = 0; j < LocalSize; j++) {
            float raf, rbf, rcf, rdf;
            int rai, rbi, rci, rdi;

            raf = in_float_local_acc[j];
            rbf = in_float_local_acc[LocalSize - 1 - j];
            rcf = in_float_local_acc[j];
            rdf = in_float_local_acc[LocalSize - 1 - j];

            rai = in_int_local_acc[j];
            rbi = in_int_local_acc[LocalSize - 1 - j];
            rci = in_int_local_acc[j];
            rdi = in_int_local_acc[LocalSize - 1 - j];

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

            out_acc1[gid] = raf + rai;
            out_acc2[gid] = rbf + rbi;
            out_acc3[gid] = rcf + rci;
            out_acc4[gid] = rdf + rdi;
          }
        }
      });
    }));
  }

  bool verify(VerificationSetting& ver) { return true; }

  static std::string getBenchmarkName() {
    std::string name = "ArithLocalMixed_";
    name.append(std::to_string(LocalSize))
        .append("_")
        .append(std::to_string(AddPercFloat))
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
};

// 131072 20000
int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<ArithLocalMixed<8, 1, 1, 1, 1, 1, 1, 0>>();
  app.run<ArithLocalMixed<16, 1, 1, 1, 1, 1, 1, 0>>();
  app.run<ArithLocalMixed<32, 1, 1, 1, 1, 1, 1, 0>>();

  app.run<ArithLocalMixed<8, 3, 3, 3, 1, 1, 1, 0>>();
  app.run<ArithLocalMixed<16, 3, 3, 3, 1, 1, 1, 0>>();
  app.run<ArithLocalMixed<32, 3, 3, 3, 1, 1, 1, 0>>();

  app.run<ArithLocalMixed<8, 1, 1, 1, 3, 3, 3, 0>>();
  app.run<ArithLocalMixed<16, 1, 1, 1, 3, 3, 3, 0>>();
  app.run<ArithLocalMixed<32, 1, 1, 1, 3, 3, 3, 0>>();

  app.run<ArithLocalMixed<8, 1, 1, 1, 1, 1, 1, 1>>();
  app.run<ArithLocalMixed<16, 1, 1, 1, 1, 1, 1, 1>>();
  app.run<ArithLocalMixed<32, 1, 1, 1, 1, 1, 1, 1>>();

  app.run<ArithLocalMixed<8, 3, 3, 3, 1, 1, 1, 1>>();
  app.run<ArithLocalMixed<16, 3, 3, 3, 1, 1, 1, 1>>();
  app.run<ArithLocalMixed<32, 3, 3, 3, 1, 1, 1, 1>>();

  app.run<ArithLocalMixed<8, 1, 1, 1, 3, 3, 3, 1>>();
  app.run<ArithLocalMixed<16, 1, 1, 1, 3, 3, 3, 1>>();
  app.run<ArithLocalMixed<32, 1, 1, 1, 3, 3, 3, 1>>();

  return 0;
}