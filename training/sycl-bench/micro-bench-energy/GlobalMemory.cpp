#include "common.h"
#include <iostream>

using namespace sycl;

template <typename ValueType>
class GlobalMemory {
protected:
  size_t size;
  size_t iters;
  std::vector<ValueType> a;
  std::vector<ValueType> c1;
  BenchmarkArgs& args;
  PrefetchedBuffer<ValueType, 1> a_buf;
  PrefetchedBuffer<ValueType, 1> c1_buf;

public:
  GlobalMemory(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size;
    iters = args.num_iterations;
    a.resize(size);
    c1.resize(size);

    for(size_t i = 0; i < size; i++) {
      a[i] = (float)(i % 1) + 1;
    }

    a_buf.initialize(args.device_queue, a.data(), sycl::range<1>{size});
    c1_buf.initialize(args.device_queue, c1.data(), sycl::range<1>{size});
  }

  void run(std::vector<sycl::event>& events) {
    // Launch the computation
    event e = args.device_queue.submit([&](handler& h) {
      auto a_acc = a_buf.template get_access<sycl::access_mode::read_write>(h);
      auto c1_acc = c1_buf.template get_access<sycl::access_mode::read_write>(h);

      range<1> grid{size};

      h.parallel_for(grid, [=, _size = size, compute_iters = iters](sycl::id<1> id) {
        constexpr size_t unrolls = 32;
        constexpr size_t stride = 32 * 1024; // size must be at least 1MB elements
        int gid = id.get(0);
        ValueType r0;

        for(int j = 0; j < compute_iters; j += unrolls) {
#pragma unroll
          for(int i = 0; i < unrolls; i++) {
            r0 = a_acc[(gid + stride * i) % _size];
            c1_acc[(gid + stride * i) % _size] = r0;
          }
        }
      }); // end parallel for
    });   // end submit

    events.push_back(e);
  }


  static std::string getBenchmarkName() {
    std::string name = "GlobalMemory_";
    name.append(std::is_same_v<ValueType, int> ? "int" : "float");
    return name;
  }

  bool verify(VerificationSetting& ver) { return true; }
};

// 1048576 1000000
int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<GlobalMemory<float>>();
}
