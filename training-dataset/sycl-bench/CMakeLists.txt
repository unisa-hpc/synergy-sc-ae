cmake_minimum_required(VERSION 3.5)
project(sycl-bench)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake Build Type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Due to CMake limitations, hipSYCL requires C++ standard to be set manually
set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -std=c++17")

if(CMAKE_GENERATOR STREQUAL "Ninja")
  set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -fdiagnostics-color=always")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
  set(COMPUTECPP_USER_FLAGS "${COMPUTECPP_USER_FLAGS} -fdiagnostics-color=always")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/polybench/common)

set(SYCL_IMPL "" CACHE STRING "Select SYCL implementation [hipSYCL | LLVM | LLVM-CUDA | triSYCL]")
option(ENABLED_SYNERGY "Enable SYnergy support" OFF)
option(ENABLED_TIME_EVENT_PROFILING "Enable time profiling with sycl event" OFF)

set(supported_implementations
  hipSYCL
  LLVM
  LLVM-CUDA
  triSYCL
)

list(FIND supported_implementations ${SYCL_IMPL} impl_idx)

if(NOT SYCL_IMPL OR impl_idx EQUAL -1)
  message(FATAL_ERROR "Please specify SYCL_IMPL (one of: ${supported_implementations})")
endif()

if(SYCL_IMPL STREQUAL "hipSYCL")
  find_package(hipSYCL CONFIG REQUIRED)
elseif(SYCL_IMPL STREQUAL "LLVM")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
elseif(SYCL_IMPL STREQUAL "LLVM-CUDA")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda")
  set(SYCL_BENCH_CUDA_ARCH "" CACHE STRING "CUDA architecture in format \"sm_xy\"")

  if(NOT SYCL_BENCH_CUDA_ARCH STREQUAL "")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend --cuda-gpu-arch=${SYCL_BENCH_CUDA_ARCH}")
  endif()
endif()

set(benchmarks
  micro-bench-energy/ArithLocalMixed.cpp
  micro-bench-energy/ArithMixedUnitOp.cpp
  micro-bench-energy/ArithMixedUnitType.cpp
  micro-bench-energy/ArithSingleUnit.cpp
  micro-bench-energy/GlobalMemory.cpp
  micro-bench-energy/GlobalMemory2.cpp
  micro-bench-energy/L2Unit.cpp
  micro-bench-energy/LocalMemory.cpp
  micro-bench-energy/Stencil.cpp
)

foreach(benchmark IN LISTS benchmarks)
  get_filename_component(target ${benchmark} NAME_WE)

  add_executable(${target} ${benchmark})

  if(ENABLED_SYNERGY)
    target_compile_definitions(${target} PUBLIC __ENABLED_SYNERGY)
    target_link_libraries(${target} PUBLIC synergy)
  endif()

  if(ENABLED_TIME_EVENT_PROFILING)
    target_compile_definitions(${target} PUBLIC SYCL_BENCH_ENABLE_QUEUE_PROFILING=1)
  endif()

  if(SYCL_IMPL STREQUAL "hipSYCL")
    add_sycl_to_target(TARGET ${target} SOURCES ${benchmark})
  endif()

  if(SYCL_IMPL STREQUAL "LLVM")
    target_compile_definitions(${target} PRIVATE __LLVM_SYCL__)
  endif()

  if(SYCL_IMPL STREQUAL "LLVM-CUDA")
    target_compile_definitions(${target} PRIVATE __LLVM_SYCL_CUDA__)
  endif()

  install(TARGETS ${target} RUNTIME DESTINATION bin/benchmarks/)
  get_filename_component(dir ${benchmark} DIRECTORY)
  set_property(TARGET ${target} PROPERTY FOLDER ${dir})
endforeach(benchmark)

# The "compiletime" target should only be used in the context of the compile time evaluation script
# set_target_properties(compiletime PROPERTIES EXCLUDE_FROM_ALL 1)
if(ENABLED_SYNERGY)
  add_subdirectory(SYnergy)
endif()

# add_subdirectory(micro-bench-energy)
install(PROGRAMS bin/run-suite DESTINATION bin/)
install(FILES ${PROJECT_SOURCE_DIR}/Brommy.bmp DESTINATION share/)
