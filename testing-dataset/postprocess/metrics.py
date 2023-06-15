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
                
    # Carica i file CSV in un DataFrame
    df = pd.read_csv(work_dir+"/"+file)
    mean_energies=df['mean-energy [J]'].values
    max_energies=df['max-energy [J]'].values
    
    times=df['run-time [s]'].values
    # edp and ed2p computation
    mean_edp = times*mean_energies
    mean_ed2p= times*times*mean_energies

    max_edp = times*max_energies
    max_ed2p= times*times*max_energies
    
    # Add metric to data frame
    df.insert(loc=len(df.columns), column='mean-edp', value=mean_edp)
    df.insert(loc=len(df.columns), column='mean-ed2p', value=mean_ed2p)
    
    df.insert(loc=len(df.columns), column='max-edp', value=max_edp)
    df.insert(loc=len(df.columns), column='max-ed2p', value=max_ed2p)
    
    out_file = file.replace(".csv", "")
    df.to_csv(f"{out_dir}/{out_file}_energy_metrics.csv", index=False,  float_format='%.8f')