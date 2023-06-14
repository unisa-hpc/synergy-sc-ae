import sys
import os
import pandas as pd
import numpy as np
from paretoset import paretoset

if len(sys.argv) != 5:
  print("<kernel_dir> <output_dir> <default_memory_freq> <default_core_freq>")
  exit(0)

kernel_dir = sys.argv[1]
output_dir = sys.argv[2]

default_memory_freq = int(sys.argv[3])
default_core_freq = int(sys.argv[4])


if not os.path.exists(output_dir):
  os.makedirs(output_dir)


with open(f"{output_dir}/ES_PL_metrics_freq_measurement.csv", "w") as out_file:
  out_file.write("kernel,es25,es50,es75,pl25,pl50,pl75\n")

  for file in os.listdir(kernel_dir):
    df = pd.read_csv(f"{kernel_dir}/{file}")
    kernel_names = df["kernel-name"]
    i=0
    first_kernel=""

    for kernel_name in kernel_names:
      if i==0:
          first_kernel = kernel_name
      if i>0 and kernel_name == first_kernel:
          break
      i+=+1

      out_file.write(f"{kernel_name}")

      kernel_data = df[df["kernel-name"] == kernel_name].copy()

      baseline_row = kernel_data[(kernel_data["core-freq"] == default_core_freq) & (kernel_data["memory-freq"] == default_memory_freq)]
      baseline_time = baseline_row["kernel-time [s]"].iloc[0]
      baseline_energy = baseline_row["max-energy [J]"].iloc[0]

      # select the appropriate kernel and compute speedup and normalized energy
      kernel_data["speedup"] = kernel_data["kernel-time [s]"].apply(lambda x: baseline_time / x)
      kernel_data["norm_energy"] = kernel_data["max-energy [J]"].apply(lambda x: x / baseline_energy)

      # filter data: only take data with a normalized energy consumption higher than two or speedup higher than 1
      kernel_data = kernel_data[(kernel_data["speedup"] > 1) | (kernel_data["max-energy [J]"] < 2*baseline_energy)]
      kernel_core_freq = kernel_data["core-freq"]
      kernel_speedup = kernel_data["speedup"]
      kernel_norm_energy = kernel_data["norm_energy"]

          
      # Take the min and max core frequency in order to set the color bar dimension
      min_core_freq=min(kernel_data["core-freq"].values)
      max_core_freq=max(kernel_data["core-freq"].values)


      df_speedup_energy = pd.DataFrame({"speedup": kernel_speedup, "energy": kernel_norm_energy})
      mask = paretoset(df_speedup_energy, sense=["max", "min"])
      pset = df_speedup_energy[mask]
      pset = pset.sort_values(by=["speedup"])

      

      min_energy_freq_row = kernel_data[kernel_data["max-energy [J]"] == min(kernel_data["max-energy [J]"].values)]

      for val in [25, 50, 75]:
        out_file.write(f",")
        perc = val / 100
        range = abs(baseline_energy - min_energy_freq_row["max-energy [J]"].iloc[0])
        energy_saving = (baseline_energy - (range * perc)) / baseline_energy
        
        argmin = (np.abs(pset["energy"] - energy_saving)).argmin()
        nearest_norm_energy = pset["energy"].iloc[argmin]
        freq = kernel_data[kernel_data["norm_energy"] == nearest_norm_energy]["core-freq"].iloc[0]
        out_file.write(f"{freq}")

      for val in [25, 50, 75]:
        out_file.write(f",")
        perc = val / 100
        range = abs(baseline_time - min_energy_freq_row["kernel-time [s]"].iloc[0])
        perf_loss = baseline_time / (baseline_time + (range * perc))

        argmin = (np.abs(pset["speedup"] - perf_loss)).argmin()
        nearest_speedup = pset["speedup"].iloc[argmin]
        freq = kernel_data[kernel_data["speedup"] == nearest_speedup]["core-freq"].iloc[0]
        out_file.write(f"{freq}")
      out_file.write("\n")
 