// Example SYCL application to demo Makefile.
#include <CL/sycl.hpp>

#include <iostream>

int main() {
  synergy::queueq;

  q.submit([&](cl::sycl::handler& cgh) {
    cl::sycl::stream ostream(1024, 64, cgh);
    cgh.single_task<class hello>([=]() { ostream << "Hello, ComputeCpp\n"; });
  });
}
