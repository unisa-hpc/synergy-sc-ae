#!/usr/bin/python3

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
if len(sys.argv) != 4:
    print("Insert path to kernel data with all frequences, edp output folder and ed2p output folder")
    exit(0)

kernel_dir = sys.argv[1]
edp_output_dir = sys.argv[2]
ed2p_output_dir = sys.argv[3]

if not os.path.exists(edp_output_dir):
    os.makedirs(edp_output_dir)
    
if not os.path.exists(ed2p_output_dir):
    os.makedirs(ed2p_output_dir)

ticks_size=11
axis_label_size=13
scatter_size=15
legend_size=11

default_core_freq = 1312
default_memory_freq = 877
pd.set_option('display.width', 1000)
for file in os.listdir(kernel_dir):
    df = pd.read_csv(kernel_dir+"/"+file)
    kernel_names = df['kernel-name']
    i=0
    first_kernel=""
    for kernel_name in kernel_names:
        if i==0:
            first_kernel = kernel_name
        if i>0 and kernel_name == first_kernel:
            break
        
        filtered_df = df[df["core-freq"] > 800]
        base_line_row = df[(df["core-freq"] == default_core_freq) & (df["memory-freq"] == default_memory_freq) & (df["kernel-name"]== kernel_name)]

        kernel_data = filtered_df[filtered_df["kernel-name"] == kernel_name]
        kernel_times = kernel_data['kernel-time [s]']
        kernel_core_freq = kernel_data['core-freq']
        kernel_memory_freq = kernel_data['memory-freq']
        kernel_max_energy = kernel_data['max-energy [J]']
        kernel_edp = kernel_data['max-edp']
        kernel_ed2p = kernel_data['max-ed2p']

        # Clear the plot to avoid that data of the previous itereation are rewritten in the plot
        plt.clf()
        plt.grid(zorder=0)
        plt.xticks(size=ticks_size)
        plt.yticks(size=ticks_size)

        plt.xlabel("Core Frequency", size=axis_label_size)
        plt.ylabel("EDP", size=axis_label_size)
        sc = plt.scatter(kernel_core_freq.values, kernel_edp.values, s=scatter_size, zorder=2)        
        if default_core_freq == 1312:
            plt.scatter(1312, base_line_row['max-edp'].values, marker='x', color='black', s=scatter_size, zorder=4, label="default configuration")
        plt.legend(fontsize=legend_size)
        plt.savefig(edp_output_dir+"/"+kernel_name+"_edp.pdf", bbox_inches='tight')
        
        # print ed2p plot
        plt.clf()
        plt.grid(zorder=0)
        plt.xticks(size=ticks_size)
        plt.yticks(size=ticks_size)

        plt.xlabel("Core Frequency", size=axis_label_size)
        plt.ylabel("ED2P", size=axis_label_size)
        sc = plt.scatter(kernel_core_freq.values, kernel_ed2p.values, s=scatter_size, zorder=2)        
        if default_core_freq == 1312:
            plt.scatter(1312, base_line_row['max-ed2p'].values, marker='x', color='black', s=scatter_size, zorder=4, label="default configuration")
        plt.legend(fontsize=legend_size)
        plt.savefig(ed2p_output_dir+"/"+kernel_name+"_ed2p.pdf", bbox_inches='tight')
        i+=+1
        
        