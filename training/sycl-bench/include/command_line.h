#ifndef BENCHMARK_COMMAND_LINE_HPP
#define BENCHMARK_COMMAND_LINE_HPP

#if defined(SYCL_BENCH_ENABLE_QUEUE_PROFILING) || defined(__ENABLED_SYNERGY)
#define queueProperties                                                                                                \
  (sycl::property_list{sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}})
#else
#define queueProperties ({})
#endif

#include "queue_macro.h"
#include "result_consumer.h"
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using CommandLineArguments = std::unordered_map<std::string, std::string>;
using FlagList = std::unordered_set<std::string>;

namespace detail {

template <class T>
inline T simple_cast(const std::string& s) {
  std::stringstream sstr{s};
  T result;
  sstr >> result;
  return result;
}

template <class T>
inline std::vector<T> parseCommaDelimitedList(const std::string& s) {
  std::stringstream istr(s);
  std::string current;
  std::vector<T> result;

  while(std::getline(istr, current, ',')) result.push_back(simple_cast<T>(current));

  return result;
}

template <class SyclArraylike>
inline SyclArraylike parseSyclArray(const std::string& s, std::size_t defaultValue) {
  auto elements = parseCommaDelimitedList<std::size_t>(s);
  if(s.size() > 3)
    throw std::invalid_argument{"Invalid sycl range/id: " + s};
  else if(s.size() == 3)
    return SyclArraylike{elements[0], elements[1], elements[2]};
  else if(s.size() == 2)
    return SyclArraylike{elements[0], elements[1], defaultValue};
  else if(s.size() == 1)
    return SyclArraylike{elements[0], defaultValue, defaultValue};
  else
    throw std::invalid_argument{"Invalid sycl range/id: " + s};
}

} // namespace detail

template <class T>
inline T cast(const std::string& s) {
  return detail::simple_cast<T>(s);
}

template <>
inline sycl::range<3> cast(const std::string& s) {
  return detail::parseSyclArray<sycl::range<3>>(s, 1);
}

template <>
inline sycl::id<3> cast(const std::string& s) {
  return detail::parseSyclArray<sycl::id<3>>(s, 0);
}

class CommandLine {
public:
  CommandLine() = default;

  CommandLine(int argc, char** argv) {
    for(int i = 0; i < argc; ++i) {
      std::string arg = argv[i];
      auto pos = arg.find("=");
      if(pos != std::string::npos) {
        auto argName = arg.substr(0, pos);
        auto argVal = arg.substr(pos + 1);

        if(args.find(argName) != args.end()) {
          throw std::invalid_argument{"Encountered command line argument several times: " + argName};
        }

        args[argName] = argVal;
      } else {
        flags.insert(arg);
      }
    }
  }

  bool isArgSet(const std::string& arg) const { return args.find(arg) != args.end(); }

  template <class T>
  T getOrDefault(const std::string& arg, const T& defaultVal) const {
    if(isArgSet(arg))
      return cast<T>(args.at(arg));
    return defaultVal;
  }

  template <class T>
  T get(const std::string& arg) const {
    try {
      return cast<T>(args.at(arg));
    } catch(std::out_of_range& e) {
      throw std::invalid_argument{"Command line argument was requested but missing: " + arg};
    }
  }

  bool isFlagSet(const std::string& flag) const { return flags.find(flag) != flags.end(); }


private:
  CommandLineArguments args;
  FlagList flags;
};


struct VerificationSetting {
  bool enabled;
  sycl::id<3> begin = {0, 0, 0};
  sycl::range<3> range = {1, 1, 1};
};

struct BenchmarkArgs {
  size_t problem_size;
#ifdef __ENABLED_SYNERGY
  synergy::frequency core_freq;
  synergy::frequency memory_freq;
#endif
  size_t num_iterations;
  size_t local_size;
  size_t num_runs;
  selected_queue device_queue;
  VerificationSetting verification;
  // can be used to query additional benchmark specific information from the command line
  CommandLine cli;
  std::shared_ptr<ResultConsumer> result_consumer;
};


int CUDASelector(const sycl::device& device) {
  using namespace sycl::info;
  const std::string driverVersion = device.get_info<device::driver_version>();
  if(device.is_gpu() && (driverVersion.find("CUDA") != std::string::npos)) {
    return 1;
  };
  return -1;
}


class BenchmarkCommandLine {
public:
  BenchmarkCommandLine(int argc, char** argv) : cli_parser{argc, argv} {}

  BenchmarkArgs getBenchmarkArgs() const {
    std::size_t size = cli_parser.getOrDefault<std::size_t>("--size", 3072);
    std::size_t local_size = cli_parser.getOrDefault<std::size_t>("--local", 256);
    std::size_t num_runs = cli_parser.getOrDefault<std::size_t>("--num-runs", 5);
#ifdef __ENABLED_SYNERGY
    synergy::frequency core_freq = cli_parser.getOrDefault<synergy::frequency>("--core-freq", 0);
    synergy::frequency memory_freq = cli_parser.getOrDefault<synergy::frequency>("--memory-freq", 0);
#endif

    std::size_t num_iterations = cli_parser.getOrDefault<std::size_t>("--num-iters", 1);
    std::string device_type = cli_parser.getOrDefault<std::string>("--device", "default");

    selected_queue q = getQueue(device_type);
#ifdef __ENABLED_SYNERGY
    q.set_target_frequencies(memory_freq, core_freq);
#endif

    bool verification_enabled = true;
    if(cli_parser.isFlagSet("--no-verification"))
      verification_enabled = false;


    auto verification_begin = cli_parser.getOrDefault<sycl::id<3>>("--verification-begin", sycl::id<3>{0, 0, 0});

    auto verification_range = cli_parser.getOrDefault<sycl::range<3>>("--verification-range", sycl::range<3>{1, 1, 1});

    auto result_consumer = getResultConsumer(cli_parser.getOrDefault<std::string>("--output", "stdio"));
#ifdef __ENABLED_SYNERGY
    return BenchmarkArgs{size, core_freq, memory_freq, num_iterations, local_size, num_runs, q,
        VerificationSetting{verification_enabled, verification_begin, verification_range}, cli_parser, result_consumer};
#else
    return BenchmarkArgs{size, num_iterations, local_size, num_runs, q,
        VerificationSetting{verification_enabled, verification_begin, verification_range}, cli_parser, result_consumer};
#endif
  }

private:
  std::shared_ptr<ResultConsumer>

  getResultConsumer(const std::string& result_consumer_name) const {
    if(result_consumer_name == "stdio")
      return std::shared_ptr<ResultConsumer>{new OstreamResultConsumer{std::cout}};
    else
      // create result consumer that appends to a csv file, interpreting the output name
      // as the target file name
      return std::shared_ptr<ResultConsumer>{new AppendingCsvResultConsumer{result_consumer_name}};
  }
  selected_queue getQueue(const std::string& device_type) const {
#if defined(__LLVM_SYCL_CUDA__)
    if(device_type != "gpu") {
      throw std::invalid_argument{"Only the 'gpu' device is supported on LLVM CUDA"};
    }

    return selected_queue(CUDASelector, queueProperties);
#endif
#ifndef __ENABLED_SYNERGY
    if(device_type == "cpu") {
      return sycl::queue{sycl::cpu_selector_v, queueProperties};
    } else if(device_type == "gpu") {
      return sycl::queue{sycl::gpu_selector_v, queueProperties};
    } else if(device_type == "default") {
      return sycl::queue{queueProperties};
    } else {
      throw std::invalid_argument{"unknown device type: " + device_type};
    }
#else
    if(device_type == "gpu") {
      return selected_queue{queueProperties};
    } else if(device_type == "default") {
      return selected_queue{queueProperties};
    } else {
      throw std::invalid_argument{"unknown device type: " + device_type};
    }
#endif
  }

  CommandLine cli_parser;
};

#endif
