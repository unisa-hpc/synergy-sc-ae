#include "common.h"
#include <iostream>

using namespace sycl;

template <typename ValueType, int radius, size_t AddPerc, size_t MulPerc, size_t DivPerc, size_t SpecPerc>
class Stencil {
protected:
  size_t size;
  size_t iters;

  std::vector<ValueType> a;
  std::vector<ValueType> b;
  std::vector<ValueType> c1;

  BenchmarkArgs& args;

  PrefetchedBuffer<ValueType, 2> a_buf;
  PrefetchedBuffer<ValueType, 2> b_buf;
  PrefetchedBuffer<ValueType, 2> c1_buf;

public:
  Stencil(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size;
    iters = args.num_iterations;
    a.resize(size * size);
    b.resize(size * size);

    c1.resize(size * size);

    for(size_t i = 0; i < size * size; i++) {
      a[i] = (ValueType)(i % 1) + 1;
      b[i] = (ValueType)(i % 1) + 1;
    }

    a_buf.initialize(args.device_queue, a.data(), sycl::range<2>{size, size});
    b_buf.initialize(args.device_queue, b.data(), sycl::range<2>{size, size});

    c1_buf.initialize(args.device_queue, c1.data(), sycl::range<2>{size, size});
  }

  void run(std::vector<sycl::event>& events) {
    // Launch the computation
    event e = args.device_queue.submit([&](handler& h) {
      auto a_acc = a_buf.template get_access<sycl::access_mode::read_write>(h);
      auto b_acc = b_buf.template get_access<sycl::access_mode::read_write>(h);

      auto c1_acc = c1_buf.template get_access<sycl::access_mode::read_write>(h);

      range<2> grid{size, size};

      h.parallel_for(grid, [=, _size = size, compute_iters = iters](sycl::id<2> id) {
        int gidx = id.get(0);
        int gidy = id.get(1);

        for(int j = 0; j < compute_iters; j++) {
          for(int x = -radius; x < radius + 1; x++)
            for(int y = -radius; y < radius + 1; y++)
              if(gidx + x > -1 && gidx + x < _size && gidy + y > -1 && gidy + y < _size) {

#pragma unroll
                for(int i = 0; i < AddPerc; i++)
                  c1_acc[gidx][gidy] += a_acc[gidx + x][gidy + y] + b_acc[gidx + x][gidy + y];
#pragma unroll
                for(int i = 0; i < MulPerc; i++) c1_acc[gidx][gidy] *= a_acc[gidx + x][gidy + y];

#pragma unroll
                for(int i = 0; i < DivPerc; i++) c1_acc[gidx][gidy] /= b_acc[gidx + x][gidy + y];

#pragma unroll
                for(int i = 0; i < SpecPerc; i++) c1_acc[gidx][gidy] = log(c1_acc[gidx][gidy]);
              }
        }
      }); // end parallel for
    });   // end submit

    events.push_back(e);
  }


  static std::string getBenchmarkName() {
    std::string name = "Stencil_";
    name.append(std::is_same_v<ValueType, int> ? "int" : "float")
        .append("_")
        .append(std::to_string(radius))
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

// 1048576 1000000
int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<Stencil<float, 2, 1, 1, 1, 0>>();
  app.run<Stencil<float, 3, 1, 1, 1, 0>>();
  app.run<Stencil<float, 4, 1, 1, 1, 0>>();
  app.run<Stencil<float, 5, 0, 1, 1, 0>>();

  app.run<Stencil<int, 2, 1, 1, 1, 0>>();
  app.run<Stencil<int, 3, 1, 1, 1, 0>>();
  app.run<Stencil<int, 4, 1, 1, 1, 0>>();
  app.run<Stencil<int, 5, 0, 1, 1, 0>>();

  app.run<Stencil<float, 2, 1, 1, 1, 1>>();
  app.run<Stencil<float, 3, 1, 1, 1, 1>>();
  app.run<Stencil<float, 4, 1, 1, 1, 1>>();
  app.run<Stencil<float, 5, 1, 0, 0, 1>>();

  app.run<Stencil<int, 2, 1, 1, 1, 1>>();
  app.run<Stencil<int, 3, 1, 1, 1, 1>>();
  app.run<Stencil<int, 4, 1, 1, 1, 1>>();
  app.run<Stencil<int, 5, 1, 0, 0, 1>>();
}
