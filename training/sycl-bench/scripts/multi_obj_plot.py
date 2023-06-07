#!/usr/bin/python3

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from paretoset import paretoset

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
if len(sys.argv) != 3:
    print("Insert path to kernel data with all frequences and output folder")
    exit(0)

kernel_dir = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ticks_size=11
axis_label_size=13
scatter_size=15
legend_size=11

default_core_freq = 1312
default_memory_freq = 877

pd.set_option("display.width", 1000)
for file in os.listdir(kernel_dir):
    df = pd.read_csv(kernel_dir+"/"+file)
    kernel_names = df["kernel-name"]
    i=0
    first_kernel=""
    for kernel_name in kernel_names:
        if i==0:
            first_kernel = kernel_name
        if i>0 and kernel_name == first_kernel:
            break
        i+=+1

        # take the row corresponding to the default frequency configuration
        baseline_row = df[(df["core-freq"] == default_core_freq) & (df["memory-freq"] == default_memory_freq) & (df["kernel-name"]== kernel_name)]
        baseline_time = baseline_row["kernel-time [s]"].iloc[0]
        baseline_energy = baseline_row["max-energy [J]"].iloc[0]

        # select the appropriate kernel and compute speedup and normalized energy
        kernel_data = df[df["kernel-name"] == kernel_name].copy()
        kernel_data["speedup"] = kernel_data["kernel-time [s]"].apply(lambda x: baseline_time / x)
        kernel_data["norm_energy"] = kernel_data["max-energy [J]"].apply(lambda x: x / baseline_energy)

        # filter data: only take data with a normalized energy consumption higher than two or speedup higher than 1
        kernel_data = kernel_data[(kernel_data["speedup"] > 1) | (kernel_data["max-energy [J]"] < 2*baseline_energy)]
        kernel_core_freq = kernel_data["core-freq"]
        kernel_speedup = kernel_data["speedup"]
        kernel_norm_energy = kernel_data["norm_energy"]

        # Clear the plot to avoid that data of the previous itereation are rewritten in the plot
        plt.clf()
       
        # Take the min and max core frequency in order to set the color bar dimension
        min_core_freq=min(kernel_data["core-freq"].values)
        max_core_freq=max(kernel_data["core-freq"].values)
       
        # create a standard color map
        cm = plt.get_cmap("viridis", max_core_freq)
        # assign a core frequency value associated to the color map for each point 
        z=kernel_core_freq.values

        plt.grid(zorder=0)

        plt.xlabel("Speedup", size=axis_label_size)
        plt.ylabel("Normalized Energy", size=axis_label_size)
        plt.xticks(size=ticks_size)
        plt.yticks(size=ticks_size)

        sc = plt.scatter(kernel_speedup.values, kernel_norm_energy.values, s=scatter_size, c=z, vmin=min_core_freq, vmax=max_core_freq, cmap=cm, zorder=2)        
        plt.scatter(1,1, marker="x", color="black", s=scatter_size, zorder=4, label="default configuration")

        color_bar=plt.colorbar(sc)
        color_bar.set_label("Core Frequency", size=axis_label_size)
        color_bar.ax.tick_params(labelsize=ticks_size)
       
        # Compute the pareto set point and print on the plot
        # Creaete a data frame with energy and speedup
        df_speedup_energy = pd.DataFrame({"speedup": kernel_speedup, "energy": kernel_norm_energy})
        mask = paretoset(df_speedup_energy, sense=["max", "min"])
        pset = df_speedup_energy[mask]
        pset = pset.sort_values(by=["speedup"])
        
        np_array = pset.to_numpy()
        pset_size = len(pset["speedup"])
        
        cur_xlim_left, cur_xlim_right = plt.xlim()
        cur_xlim_bottom, cur_ylim_top = plt.ylim()
        x1, y1 = [cur_xlim_left, np_array[0][0]], [np_array[0][1], np_array[0][1]]
        plt.plot(x1, y1, color="red", linewidth=2.5, label="Pareto Front")

        for i in range(pset_size):
            if not (i == pset_size-1):
                current_x = np_array[i][0]
                current_y = np_array[i][1]
                next_x = np_array[i+1][0]
                next_y = np_array[i+1][1]
                x1, y1 = [current_x, current_x], [current_y, next_y]
                x2, y2 = [current_x, next_x], [next_y, next_y]
                plt.plot(x1, y1, x2, y2, color="red", linewidth=2.5)

        last_point = np_array[pset_size-1]
        x1, y1 = [last_point[0], last_point[0]], [last_point[1], cur_ylim_top]
        plt.plot(x1, y1, color="red", linewidth=2.5)


        plt.xlim(left=cur_xlim_left)
        plt.ylim(top=cur_ylim_top)
        plt.legend(fontsize=legend_size)
        plt.savefig(output_dir+"/"+kernel_name+".pdf", bbox_inches="tight")

        if kernel_name == "BlackScholes":
            hlines = []
            line_style = [":", "-.", "--", "-"] 
            xmin, xmax = plt.xlim()
            for index, val in enumerate([1, 5, 10, 20]):
                perc = val / 100
                energy_saving = (baseline_energy - (baseline_energy * perc)) / baseline_energy
                hlines.append(plt.axhline(energy_saving, label="Energy Saving "+str(val)+"%", color="tab:blue", linestyle=line_style[index]))
                plt.legend(fontsize=legend_size)

            color_bar.remove()
            color_bar.set_label("", size=axis_label_size)
            plt.savefig(output_dir+"/"+kernel_name+"_es.pdf", bbox_inches="tight")

            for line in hlines: line.remove()
            for index, val in enumerate([1, 5, 10, 20]):
                perc = val / 100
                perf_loss = baseline_time / (baseline_time + (baseline_time * perc))
                plt.axvline(perf_loss, label="Performance Loss "+str(val)+"%", color="tab:orange", linestyle=line_style[index])
                plt.legend(fontsize=legend_size)
                
            color_bar = plt.colorbar(sc)
            color_bar.set_label("Core Frequency", size=axis_label_size)
            plt.ylabel("")
            plt.savefig(output_dir+"/"+kernel_name+"_pl.pdf", bbox_inches="tight")

