#include "common.h"
#include <iostream>

using namespace sycl;


template <typename ValueType, size_t LocalSize>
class LocalMemory {
protected:
  size_t size;
  size_t iters;
  BenchmarkArgs& args;

public:
  LocalMemory(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size;
    iters = args.num_iterations;
  }

  void run(std::vector<sycl::event>& events) {
    // Launch the computation
    sycl::event e = args.device_queue.submit([&](handler& h) {
      sycl::local_accessor<ValueType, 1> local_acc1{LocalSize, h};

      sycl::nd_range<1> ndr{size, LocalSize};
      h.parallel_for(ndr, [=, _size = size, compute_iters = iters](sycl::nd_item<1> item) {
        int gid = item.get_global_id(0);
        int lid = item.get_local_id(0);
        ValueType r0;

        if(gid < _size) {
          for(int i = 0; i < compute_iters; i++) {
            r0 = local_acc1[lid];
            local_acc1[LocalSize - lid - 1] = r0;
          }
        }
      }); // end parallel_for
    });   // end submit
    events.push_back(e);
  }


  static std::string getBenchmarkName() {
    std::string name = "LocalMemory_";
    name.append(std::is_same_v<ValueType, int> ? "int" : "float");
    return name;
  }

  bool verify(VerificationSetting& ver) { return true; }
};

// 1000000 1000000
int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<LocalMemory<float, 32>>();
}
