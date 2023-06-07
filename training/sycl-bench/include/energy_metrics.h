#pragma once

#include <chrono>
#include <cmath>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "benchmark_traits.h"
#include "result_consumer.h"

/**
 * Throughput metrics can be returned by benchmarks that implement the
 * getThroughputMetric() function. The returned value (and associated unit)
 * represents the metric underlying the throughput calculation associated with
 * a benchmark.
 *
 * Note that the metric is NOT the throughput. For example, a returned metric
 * for arithmetric throughput could be the total number of floating-point operations,
 * FLOP, not FLOP/s.
 */

template <typename Benchmark>
class EnergyMetricsProcessor {
public:
  EnergyMetricsProcessor(const BenchmarkArgs& args) : args(args) {}

  void addEnergyResult(const std::string& name, double time) {
    if(unavailableEnergies.count(name) != 0) {
      throw std::invalid_argument{"Cannot add result for unavailable energy " + name};
    }

    energyResults[name].push_back(time);
  }

  /**
   * This is a bit of a hack that we need right now to ensure that all emitted results include the same
   * CSV columns, even if a timing is not available for a particular benchmark and/or SYCL implementation.
   *
   * TODO: Come up with a better solution
   */
  void markAsUnavailable(const std::string& name) {
    if(energyResults.count(name) != 0) {
      throw std::invalid_argument{"Cannot mark energy " + name + " with existing results as unavailable"};
    }
    unavailableEnergies.insert(name);
  }

  void emitResults(ResultConsumer& consumer) const {
    // We have to ensure that available and unavailable timings are always being emitted in the same order.
    // To this end, we copy all timing names into a sorted container and iterate over it afterwards.
    std::set<std::string> allEnergies;
    for(const auto& name : unavailableEnergies) {
      allEnergies.insert(name);
    }
    for(const auto& [name, results] : energyResults) {
      allEnergies.insert(name);
    }

    for(const auto& name : allEnergies) {
      if(unavailableEnergies.count(name) == 0) {
        std::vector<double> resultsEnergy = energyResults.at(name);
        std::sort(resultsEnergy.begin(), resultsEnergy.end());

        double mean = std::accumulate(resultsEnergy.begin(), resultsEnergy.end(), 0.0) /
                      static_cast<double>(resultsEnergy.size());

        double stddev = 0.0;
        for(double x : resultsEnergy) {
          double dev = mean - x;
          stddev += dev * dev;
        }
        if(resultsEnergy.size() <= 1) {
          stddev = 0.0;
        } else {
          stddev /= static_cast<double>(resultsEnergy.size() - 1);
          stddev = std::sqrt(stddev);
        }

        const double median = resultsEnergy[resultsEnergy.size() / 2];

        consumer.consumeResult(name + "-mean", std::to_string(mean), "J");
        consumer.consumeResult(name + "-stddev", std::to_string(stddev), "J");
        consumer.consumeResult(name + "-median", std::to_string(median), "J");
        consumer.consumeResult(name + "-min", std::to_string(resultsEnergy[0]), "J");
        consumer.consumeResult(name + "-max", std::to_string(resultsEnergy[resultsEnergy.size()-1]), "J");


        // Emit individual samples as well
        std::stringstream samples;
        samples << "\"";
        for(int i = 0; i < resultsEnergy.size(); ++i) {
          samples << std::to_string(resultsEnergy[i]);
          if(i != resultsEnergy.size() - 1) {
            samples << " ";
          }
        }
        samples << "\"";
        consumer.consumeResult(name + "-samples", samples.str());

      } else {
        // Now the hacky part: Emit columns also for unavailable timings.
        // FIXME: Come up with a cleaner solution.
        consumer.consumeResult(name + "-mean", "N/A");
        consumer.consumeResult(name + "-stddev", "N/A");
        consumer.consumeResult(name + "-median", "N/A");
        consumer.consumeResult(name + "-min", "N/A");
        consumer.consumeResult(name + "-samples", "N/A");
        consumer.consumeResult(name + "-throughput", "N/A");
      }
    }
  }

private:
  const BenchmarkArgs args;
  std::unordered_map<std::string, std::vector<double>> energyResults;
  std::unordered_set<std::string> unavailableEnergies;
};
