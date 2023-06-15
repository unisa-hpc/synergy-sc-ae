#!/usr/bin/python3

import os
import sys
import pandas as pd

if len(sys.argv) != 3:
    print("Insert path to sycl-bench folder as command line argument")
    exit(0)

sycl_bench_csv_dir=sys.argv[1]
merged_csv_dir=sys.argv[2]

if not os.path.exists(merged_csv_dir):
    os.makedirs(merged_csv_dir)

sycl_bench_files={}
features_files={}

for sycl_bench_file in os.listdir(sycl_bench_csv_dir):
    kernel = sycl_bench_file[::-1].split('_', 5)[5][::-1]    

    if not kernel in sycl_bench_files:
        sycl_bench_files[kernel] = []

    if kernel in sycl_bench_file: # we only take the kernel name from the sycl-bench result file
        sycl_bench_files[kernel].append(sycl_bench_csv_dir+"/"+sycl_bench_file)


df_all = pd.DataFrame()
for kernel in sycl_bench_files:
    df_kernel = pd.DataFrame()

    for sbench_file in sycl_bench_files[kernel]:
        df_sbench = pd.read_csv(sbench_file)
        # remove feature kernel name
        df_kernel = pd.concat([df_kernel, df_sbench], ignore_index=True)

    df_all = pd.concat([df_all, df_kernel], ignore_index=True)
    df_kernel.to_csv(merged_csv_dir+"/merged_"+kernel+".csv" , index=False, float_format='%.8f')

# df_all.to_csv(merged_csv_dir+"/merged_all.csv" , index=False, float_format='%.8f')

    # print(df_merged)
