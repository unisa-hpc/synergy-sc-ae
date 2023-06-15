if(ENABLED_SYNERGY)
  link_libraries(synergy)
endif()


add_executable(Arithmetic Arithmetic.cpp)
add_executable(ArithmeticLocalMemory ArithmeticLocalMemory.cpp)
add_executable(GlobalMemory GlobalMemory.cpp)
add_executable(LocalMemory LocalMemory.cpp)
add_executable(ArithMixedUnitOp ArithMixedUnitOp.cpp)


add_executable(SpecialFunctions SpecialFunctions.cpp)
add_executable(GlobalMem GlobalMem.cpp)


get_directory_property(all_targets BUILDSYSTEM_TARGETS)

foreach(target IN LISTS all_targets)
    
    target_compile_definitions(${target} PUBLIC SYCL_BENCH_ENABLE_QUEUE_PROFILING=1)
    target_compile_definitions(${target} PUBLIC __HIPSYCL__)
    target_compile_definitions(${target} PUBLIC __LLVM_SYCL__)
    target_compile_definitions(${target} PUBLIC __LLVM_SYCL_CUDA__)
  
    if(ENABLED_SYNERGY)
      target_compile_definitions(${target} PUBLIC __ENABLED_SYNERGY)
    endif()
  
    if(SYCL_IMPL STREQUAL "hipSYCL")
      add_sycl_to_target(TARGET ${target} SOURCES ${benchmark})
    endif()
endforeach()