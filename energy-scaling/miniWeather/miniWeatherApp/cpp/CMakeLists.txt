cmake_minimum_required(VERSION 3.8)
project(miniWeather CXX)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

enable_testing()

############################################################
## Set Parameters
############################################################
if ("${NX}" STREQUAL "")
  SET(NX 100)
endif()
if ("${NZ}" STREQUAL "")
  SET(NZ 50)
endif()
if ("${SIM_TIME}" STREQUAL "")
  SET(SIM_TIME 1000)
endif()
if ("${OUT_FREQ}" STREQUAL "")
  SET(OUT_FREQ 10)
endif()
if ("${DATA_SPEC}" STREQUAL "")
  SET(DATA_SPEC DATA_SPEC_THERMAL)
endif()
SET(EXE_DEFS "-D_NX=${NX} -D_NZ=${NZ} -D_SIM_TIME=${SIM_TIME} -D_OUT_FREQ=${OUT_FREQ} -D_DATA_SPEC=${DATA_SPEC}")
SET(TEST_DEFS "-D_NX=100 -D_NZ=50 -D_SIM_TIME=400 -D_OUT_FREQ=400 -D_DATA_SPEC=DATA_SPEC_THERMAL")

if ( ("${YAKL_CXX_FLAGS}"    MATCHES ".*SINGLE_PREC.*") OR 
     ("${YAKL_CUDA_FLAGS}"   MATCHES ".*SINGLE_PREC.*") OR 
     ("${YAKL_HIP_FLAGS}"    MATCHES ".*SINGLE_PREC.*") OR 
     ("${YAKL_OPENMP_FLAGS}" MATCHES ".*SINGLE_PREC.*") OR 
     ("${YAKL_SYCL_FLAGS}"   MATCHES ".*SINGLE_PREC.*") )
  message(STATUS "Using single precision")
else()
  message(STATUS "Using double precision")
endif()

############################################################
## Append CXXFLAGS
############################################################
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXFLAGS}")


############################################################
## Add YAKL 
############################################################
add_subdirectory(YAKL)
if (${CMAKE_VERSION} VERSION_GREATER "3.18.0")
  set_property(TARGET yakl PROPERTY CUDA_ARCHITECTURES OFF)
endif()
include_directories(YAKL)



############################################################
## YAKL parallel_for Version
############################################################
add_executable(parallelfor miniWeather_mpi_parallelfor.cpp)
set_target_properties(parallelfor PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")

add_executable(parallelfor_test miniWeather_mpi_parallelfor.cpp)
set_target_properties(parallelfor_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS}")

target_link_libraries(parallelfor      "${LDFLAGS}")
target_link_libraries(parallelfor_test "${LDFLAGS}")

add_test(NAME YAKL_TEST COMMAND ./check_output.sh ./parallelfor_test 1e-9 4.5e-5 )

include(YAKL/yakl_utils.cmake)

yakl_process_target(parallelfor)
yakl_process_target(parallelfor_test)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
  set_target_properties(serial serial_test mpi mpi_test parallelfor parallelfor_test parallelfor_simd_x parallelfor_simd_x_test  PROPERTIES LINKER_LANGUAGE CXX)
  if (${CMAKE_VERSION} VERSION_GREATER "3.18.0")
    set_target_properties(serial serial_test mpi mpi_test parallelfor parallelfor_test parallelfor_simd_x parallelfor_simd_x_test  PROPERTIES CUDA_ARCHITECTURES OFF)
  endif()
endif()



