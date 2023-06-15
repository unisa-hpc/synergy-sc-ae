#include "common.h"

#include <iomanip>
#include <iostream>
#include <type_traits>

// using namespace sycl;
namespace s = sycl;

template <typename T, bool>
class ScalarProdKernel;
template <typename T, bool>
class ScalarProdKernelHierarchical;

template <typename T, bool>
class ScalarProdReduction;
template <typename T, bool>
class ScalarProdReductionHierarchical;
template <typename T, bool>
class ScalarProdGatherKernel;

template <typename T, bool Use_ndrange = true>
class ScalarProdBench {
protected:
  size_t num_iters;

  std::vector<T> input1;
  std::vector<T> input2;
  std::vector<T> output;
  size_t size;
  size_t local_size;
  BenchmarkArgs& args;

  PrefetchedBuffer<T, 1> input1_buf;
  PrefetchedBuffer<T, 1> input2_buf;
  PrefetchedBuffer<T, 1> output_buf;

public:
  ScalarProdBench(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    num_iters = args.num_iterations;
    size = args.problem_size;
    local_size = args.local_size;

    // host memory allocation and initialization
    input1.resize(size);
    input2.resize(size);
    output.resize(size);

    for(size_t i = 0; i < size; i++) {
      input1[i] = static_cast<T>(1);
      input2[i] = static_cast<T>(2);
      output[i] = static_cast<T>(0);
    }

    input1_buf.initialize(args.device_queue, input1.data(), s::range<1>(size));
    input2_buf.initialize(args.device_queue, input2.data(), s::range<1>(size));
    output_buf.initialize(args.device_queue, output.data(), s::range<1>(size));
  }

  void run(std::vector<sycl::event>& events) {
    events.push_back(args.device_queue.submit([&](sycl::handler& cgh) {
      auto in1 = input1_buf.template get_access<s::access::mode::read>(cgh);
      auto in2 = input2_buf.template get_access<s::access::mode::read>(cgh);
      // Use discard_write here, otherwise the content of the hostbuffer must first be copied to device
      auto intermediate_product = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      sycl::nd_range<1> ndrange(size, local_size);

      cgh.parallel_for<class ScalarProdKernel<T, Use_ndrange>>(
          ndrange, [=, num_iters = num_iters](sycl::nd_item<1> item) {
            for(size_t i = 0; i < num_iters; i++) {
              size_t gid = item.get_global_linear_id();
              intermediate_product[gid] = in1[gid] * in2[gid];
            }
          });
    }));


    auto array_size = size;
    auto wgroup_size = local_size;
    // Not yet tested with more than 2
    auto elements_per_thread = 2;

    while(array_size != 1) {
      auto n_wgroups = (array_size + wgroup_size * elements_per_thread - 1) /
                       (wgroup_size * elements_per_thread); // two threads per work item

      events.push_back(args.device_queue.submit([&](sycl::handler& cgh) {
        auto global_mem = output_buf.template get_access<s::access::mode::read_write>(cgh);

        // local memory for reduction
        auto local_mem = s::local_accessor<T, 1>{s::range<1>(wgroup_size), cgh};
        sycl::nd_range<1> ndrange(n_wgroups * wgroup_size, wgroup_size);

        //   if(Use_ndrange) {
        cgh.parallel_for<class ScalarProdReduction<T, Use_ndrange>>(
            ndrange, [=, num_iters = num_iters](sycl::nd_item<1> item) {
              size_t gid = item.get_global_linear_id();
              size_t lid = item.get_local_linear_id();

              // initialize local memory to 0
              local_mem[lid] = 0;

              for(size_t iter = 0; iter < num_iters; iter++) {
                for(int i = 0; i < elements_per_thread; ++i) {
                  int input_element = gid + i * n_wgroups * wgroup_size;

                  if(input_element < array_size)
                    local_mem[lid] += global_mem[input_element];
                }

                item.barrier(s::access::fence_space::local_space);

                for(size_t stride = wgroup_size / elements_per_thread; stride >= 1; stride /= elements_per_thread) {
                  if(lid < stride) {
                    for(int i = 0; i < elements_per_thread - 1; ++i) {
                      local_mem[lid] += local_mem[lid + stride + i];
                    }
                  }
                  item.barrier(s::access::fence_space::local_space);
                }

                // Only one work-item per work group writes to global memory
                if(lid == 0) {
                  global_mem[item.get_global_id()] = local_mem[0];
                }
              }
            });
      }));

      events.push_back(args.device_queue.submit([&](sycl::handler& cgh) {
        auto global_mem = output_buf.template get_access<s::access::mode::read_write>(cgh);

        cgh.parallel_for<ScalarProdGatherKernel<T, Use_ndrange>>(
            sycl::range<1>{n_wgroups}, [=, num_iters = num_iters](sycl::id<1> idx) {
              for(int i = 0; i < num_iters; i++) global_mem[idx] = global_mem[idx * wgroup_size];
            });
      }));
      array_size = n_wgroups;
    }
  }

  bool verify(VerificationSetting& ver) {
    bool pass = true;
    // auto expected = static_cast<T>(0);

    // auto output_acc = output_buf.get_host_access();

    // for(size_t i = 0; i < args.problem_size; i++) {
    //   expected += input1[i] * input2[i];
    // }

    // // std::cout << "Scalar product on CPU =" << expected << std::endl;
    // // std::cout << "Scalar product on Device =" << output[0] << std::endl;

    // // Todo: update to type-specific test (Template specialization?)
    // const auto tolerance = 0.00001f;
    // if(std::fabs(expected - output_acc[0]) > tolerance) {
    //   pass = false;
    // }

    return pass;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "ScalarProduct_";
    name << (Use_ndrange ? "NDRange_" : "Hierarchical_");
    name << ReadableTypename<T>::name;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  // if(app.shouldRunNDRangeKernels()) {
  //   app.run<ScalarProdBench<int, true>>();
  //   app.run<ScalarProdBench<long long, true>>();
  //   app.run<ScalarProdBench<float, true>>();
  //   app.run<ScalarProdBench<double, true>>();
  // }

  app.run<ScalarProdBench<float, true>>();
  // app.run<ScalarProdBench<long long, false>>();
  // app.run<ScalarProdBench<float, false>>();
  // app.run<ScalarProdBench<double, false>>();

  return 0;
}
