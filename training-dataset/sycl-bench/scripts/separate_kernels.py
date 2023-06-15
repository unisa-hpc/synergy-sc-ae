#!/usr/bin/python3

import os
import sys
import pandas as pd

if len(sys.argv) != 3:
    print("Insert path to sycl-bench folder as command line argument")
    exit(0)

work_dir=sys.argv[1]
out_dir=sys.argv[2]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for file in os.listdir(work_dir):  
    df = pd.read_csv(work_dir+"/"+file)
    grouped = df.groupby("kernel-name")

    for name, group in grouped:
      group.to_csv(out_dir+"/"+name+".csv", index=False,  float_format='%.8f')
    

    




