#include <iostream>
#include <sycl/sycl.hpp>

#include "common.h"




#define TILE_DIM 32
#define BLOCK_ROWS 8



using namespace sycl;

// width, height matrix dimensions
class matrix_transpose{
    private:
        const accessor<float, 1, access_mode::read> in_matrix;
        accessor<float, 1, access_mode::read_write> out_matrix;
        local_accessor<float, 2> tile;
        size_t size;
        size_t num_iters;

    public:
        matrix_transpose(
                const accessor<float, 1, access_mode::read, target::device> in_matrix,
                accessor<float, 1, access_mode::read_write, target::device> out_matrix,
                local_accessor<float, 2> tile,
                size_t size,
                size_t num_iters
        )
        :
        in_matrix(in_matrix),
        out_matrix(out_matrix),
        tile(tile),
        size(size),
        num_iters(num_iters){}
        void operator()(nd_item<2> it) const{
            const auto group = it.get_group();
            int local_id_x=it.get_local_id(1);
            int local_id_y=it.get_local_id(0);

            int block_x = it.get_group(1);
            int block_y = it.get_group(0);

            int xIndex = block_x * TILE_DIM +local_id_x;
            int yIndex = block_y * TILE_DIM + local_id_y;
            
            int index_in = xIndex + (yIndex)*size;

            xIndex = block_y * TILE_DIM + local_id_x;
            yIndex = block_x * TILE_DIM + local_id_y;
            int index_out = xIndex + (yIndex) * size;
            
            for(int iter = 0; iter< num_iters; iter++){
                // Copy data in local memory
                for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
                    tile[local_id_y+i][local_id_x] = in_matrix[index_in+i*size];
                }
                
                group_barrier(group);

                for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
                    out_matrix[index_out+i*size] = tile[local_id_x][local_id_y+i];
                }
            }
        }


};


template<class T>
class MatrixTranspose {
protected:
  size_t num_iters;
  std::vector<T> in;
  std::vector<T> out;
  
  PrefetchedBuffer<T, 1> in_buf;
  PrefetchedBuffer<T, 1> out_buf;
  
  size_t size;
  BenchmarkArgs& args;

public:
  MatrixTranspose(BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size; // input size defined by the user
    
    num_iters = args.num_iterations;

    in.resize(size*size);
    out.resize(size*size);

    std::fill(in.begin(), in.end(), 1);
    std::fill(out.begin(), out.end(), 0);
    
    in_buf.initialize(args.device_queue, in.data(), range<1>{size*size});
    out_buf.initialize(args.device_queue, out.data(), range<1>{size*size});
  }

  void run(std::vector<sycl::event>& events) {
    events.push_back(
        args.device_queue.submit([&](handler &cgh){
            range<2> grid {BLOCK_ROWS * (size / TILE_DIM), TILE_DIM * (size / TILE_DIM)}; 
            range<2> block{BLOCK_ROWS, TILE_DIM};
            auto acc_in = in_buf.template get_access<access_mode::read>(cgh);
            auto acc_out = out_buf.template get_access<access_mode::read_write>(cgh);
            local_accessor<float, 2> tile {range<2>{TILE_DIM, TILE_DIM+1}, cgh};
            cgh.parallel_for(nd_range<2>{grid, block}, matrix_transpose(acc_in, acc_out, tile, size, num_iters));//end parallel for
        })
    );// end events.push back
    
  }


  bool verify(VerificationSetting& ver) {
    out_buf.reset();
   
    return true;
  }

  static std::string getBenchmarkName() { return "Matrix_transpose"; }

}; 


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<MatrixTranspose<float>>();
  return 0;
}
