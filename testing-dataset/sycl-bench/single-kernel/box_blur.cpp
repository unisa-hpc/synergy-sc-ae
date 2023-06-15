#include <iostream>
#include <filesystem>
#include <sycl.hpp>

#include "bitmap.h"

#include "common.h"
#define RADIUS 3

using namespace sycl;

std::filesystem::path program_dir;

class box_blur{
    private:
        const accessor<sycl::float4, 2, access_mode::read> in;
        accessor<sycl::float4, 2, access_mode::read_write> out;
        const int size;
        const int num_iters;


    public:
        box_blur(
            const accessor<sycl::float4, 2, access_mode::read> in,
            accessor<sycl::float4, 2, access_mode::read_write> out,
            const int &size,
            const int &num_iters
        ):
        in(in),
        out(out),
        size(size),
        num_iters(num_iters){}

        void operator()(nd_item<2> it) const {
            
          sycl::id<2> gid = it.get_global_id();
          int x = gid[0];
          int y = gid[1];
          for(size_t i = 0; i < num_iters; i++) {

            if(x < size && y < size) {
              sycl::float4 sum_neigh{0, 0, 0, 0};
              int hits = 0;
              for(int ox = -RADIUS; ox < RADIUS + 1; ++ox) {
                for(int oy = -RADIUS; oy < RADIUS + 1; ++oy) {
                  // image boundary check
                  if((x + ox) > -1 && (x + ox) < size && (y + oy) > -1 && (y + oy) < size) {
                    sum_neigh = sum_neigh + in[x + ox][y + oy];
                    hits++;
                  }
                }
              }
              sycl::float4 mean_neigh{sum_neigh.x() / hits, sum_neigh.y() / hits, sum_neigh.z() / hits, 0};
              out[gid] = mean_neigh;
                  
            }
          }
        }
};




namespace s = sycl;
class BoxBlurBenchKernel; // kernel forward declaration

/*
  A Sobel filter with a convolution matrix 3x3.
  Input and output are two-dimensional buffers of floats.
 */
class BoxBlurBench {
protected:
  size_t num_iters;
  std::vector<sycl::float4> input;
  std::vector<sycl::float4> output;

  size_t w, h; // size of the input picture
  size_t size; // user-defined size (input and output will be size x size)
  BenchmarkArgs& args;


  PrefetchedBuffer<sycl::float4, 2> input_buf;
  PrefetchedBuffer<sycl::float4, 2> output_buf;

public:
  BoxBlurBench(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size; // input size defined by the user
    num_iters = args.num_iterations;

    input.resize(size * size);
    load_bitmap_mirrored(program_dir.string() + "../Brommy.bmp", size, input);
    output.resize(size * size);

    input_buf.initialize(args.device_queue, input.data(), s::range<2>(size, size));
    output_buf.initialize(args.device_queue, output.data(), s::range<2>(size, size));
  }

  void run(std::vector<sycl::event>& events) {
    events.push_back(args.device_queue.submit([&](sycl::handler& cgh) {
      auto in = input_buf.get_access<s::access::mode::read>(cgh);
      auto out = output_buf.get_access<s::access::mode::read_write>(cgh);
      sycl::nd_range<2> ndrange{sycl::range<2>(size,size), sycl::range<2>{16,16}};

      cgh.parallel_for<class BoxBlurBenchKernel>(
          ndrange, box_blur(in, out, size, num_iters));
    }));
  }


  bool verify(VerificationSetting& ver) {
    // Triggers writeback
    output_buf.reset();
    // save_bitmap("box_blur.bmp", size, output);

    return true;
  }


  static std::string getBenchmarkName() { return "BoxBlur"; }

}; // BoxBlurBench class


int main(int argc, char** argv) {
  program_dir = argv[0];
  program_dir.remove_filename();
  BenchmarkApp app(argc, argv);
  app.run<BoxBlurBench>();
  return 0;
}
