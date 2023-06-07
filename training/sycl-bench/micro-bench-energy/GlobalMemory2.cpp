#include "common.h"
#include <iostream>

using namespace sycl;

template <typename ValueType, size_t k>
class GlobalMemory2 {
protected:
  size_t size;
  size_t iters;
  std::vector<ValueType> a1;
  std::vector<ValueType> a2;
  std::vector<ValueType> c1;
  std::vector<ValueType> c2;
  BenchmarkArgs& args;
  PrefetchedBuffer<ValueType, 1> a1_buf;
  PrefetchedBuffer<ValueType, 1> a2_buf;
  PrefetchedBuffer<ValueType, 1> c1_buf;
  PrefetchedBuffer<ValueType, 1> c2_buf;

public:
  GlobalMemory2(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size;
    iters = args.num_iterations;
    a1.resize(size);
    a2.resize(size);

    c1.resize(size);
    c2.resize(size);

    for(size_t i = 0; i < size; i++) {
      a1[i] = (float)(i % 1) + 1;
      a2[i] = (float)(i % 1) + 1;
    }

    a1_buf.initialize(args.device_queue, a1.data(), sycl::range<1>{size});
    a2_buf.initialize(args.device_queue, a2.data(), sycl::range<1>{size});
    c1_buf.initialize(args.device_queue, c1.data(), sycl::range<1>{size});
    c2_buf.initialize(args.device_queue, c2.data(), sycl::range<1>{size});
  }

  void run(std::vector<sycl::event>& events) {
    // Launch the computation
    event e = args.device_queue.submit([&](handler& h) {
      auto a1_acc = a1_buf.template get_access<sycl::access_mode::read_write>(h);
      auto a2_acc = a2_buf.template get_access<sycl::access_mode::read_write>(h);
      auto c1_acc = c1_buf.template get_access<sycl::access_mode::read_write>(h);
      auto c2_acc = c2_buf.template get_access<sycl::access_mode::read_write>(h);

      range<1> grid{size};

      h.parallel_for(grid, [=, _size = size, compute_iters = iters](sycl::id<1> id) {
        int gid = id.get(0);
        ValueType r0, r1;

        for(int j = 0; j < compute_iters; j++) {

#pragma unroll 4
          for(size_t i = 0; i < k; i++) {
            if(gid + i < _size) {
              r0 = a1_acc[gid + i];
              r1 = a2_acc[gid + i];
              c1_acc[gid + i] = r0;
              c2_acc[gid + i] = r1;
            }
          }
        }
      }); // end parallel for
    });   // end submit

    events.push_back(e);
  }


  static std::string getBenchmarkName() {
    std::string name = "GlobalMemory2_";
    name.append(std::is_same_v<ValueType, int> ? "int" : "float").append("_").append(std::to_string(k));
    return name;
  }

  bool verify(VerificationSetting& ver) { return true; }
};

// 524288 200
int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<GlobalMemory2<float, 2048>>();
  app.run<GlobalMemory2<float, 8192>>();
  app.run<GlobalMemory2<float, 32768>>();
}
