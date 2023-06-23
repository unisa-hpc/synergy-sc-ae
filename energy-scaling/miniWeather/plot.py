#!/usr/bin/python3

import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter
import pandas as pd

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

parsed_dir = f"{script_dir}/parsed"
output_dir = f"{script_dir}/../plots"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ticks_size=11
axis_label_size=13
line_size=2
legend_size=11

def format_num(x, pos):
    return "{:.0f}".format(x)

configurations = ["es_50", "pl_50", "default", "min_edp", "min_ed2p"] # 11
markers = ["v", "d", "x", "s", "D"]
colors = ["tab:blue", "tab:green", "tab:red", "tab:olive", "tab:purple"]

gpus = [4, 8, 16, 32, 64]

for scaling in ["ws"]:
  plt.clf()
  for gpu_index, ngpus in enumerate(gpus):
    min = []
    es = []
    pl = []
    default = []
    edp = []
    max = []
    for conf in configurations:
      name = f"miniweather_{conf}_{scaling}"
      df = pd.read_csv(f"{parsed_dir}/{name}.csv")
      df = df.sort_values("ngpus")
      df["energy"] = df["energy"] / 1000
      
      if conf == "min_energy":
        min.append(df[df["ngpus"] == ngpus])
      elif "es" in conf:
         es.append(df[df["ngpus"] == ngpus])
      elif "pl" in conf:
         pl.append(df[df["ngpus"] == ngpus])
      elif conf == "default":
         default.append(df[df["ngpus"] == ngpus])
      elif "min_ed" in conf:
         edp.append(df[df["ngpus"] == ngpus])
      else:
         max.append(df[df["ngpus"] == ngpus])

    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_num))
    plt.xticks(size=ticks_size)
    plt.yticks(size=ticks_size)
    plt.grid(zorder=0)
    plt.xlabel("Time [s]", size=axis_label_size)
    plt.ylabel("Energy [kJ]", size=axis_label_size)

    marker_index = 0
    for data in min:
      plt.scatter(data["time"].values, data["energy"].values, linewidth=line_size, marker=markers[marker_index], zorder=4, color=colors[gpu_index])
      marker_index+=1

    for data in es:
      plt.scatter(data["time"].values, data["energy"].values, linewidth=line_size, marker=markers[marker_index], zorder=4, color=colors[gpu_index], facecolors='none')
      marker_index+=1

    for data in pl:
      plt.scatter(data["time"].values, data["energy"].values, linewidth=line_size, marker=markers[marker_index], zorder=4, color=colors[gpu_index], facecolors='none')
      marker_index+=1

    for data in default:
      plt.scatter(data["time"].values, data["energy"].values, linewidth=line_size, marker=markers[marker_index], zorder=4, color=colors[gpu_index])
      marker_index+=1

    for data in edp:
      plt.scatter(data["time"].values, data["energy"].values, linewidth=line_size, marker=markers[marker_index], zorder=4, color=colors[gpu_index], facecolors='none')
      marker_index+=1

    for data in max:
      plt.scatter(data["time"].values, data["energy"].values, linewidth=line_size, marker=markers[marker_index], zorder=4, s=70, color=colors[gpu_index])
      marker_index+=1

  color_handles = []
  for index, color in enumerate(colors):
     color_handles.append(mpatches.Patch(color=color, label=f"{gpus[index]} GPUs"))
  
  marker_handles = []
  for index, marker in enumerate(markers):
     marker_handles.append(mlines.Line2D([], [], color="black", marker=marker, markerfacecolor='None', linestyle='None', markersize=7,  label=f"{configurations[index].upper()}"))

  legend2 = plt.legend(handles=marker_handles, fontsize=legend_size)
  legend1 = plt.legend(handles=color_handles, fontsize=legend_size, loc="upper center")
  plt.gca().add_artist(legend2)


  plt.savefig(f"{output_dir}/miniweather_{scaling}.pdf", bbox_inches="tight")
