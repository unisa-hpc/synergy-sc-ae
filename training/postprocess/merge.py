#!/usr/bin/python3

import os
import sys
import pandas as pd

if len(sys.argv) < 4:
    print("Insert path to sycl-bench folder as command line argument")
    exit(0)

sycl_bench_csv_dir=sys.argv[1]
features_csv_dir=sys.argv[2]
merged_csv_dir=sys.argv[3]

if not os.path.exists(merged_csv_dir):
    os.makedirs(merged_csv_dir)

norm=False
if len(sys.argv) == 5 and sys.argv[4] == "norm":
    norm=True


sycl_bench_files={}
features_files={}

for features_file in os.listdir(features_csv_dir):
    kernel = features_file.replace("_features.csv", "")

    sycl_bench_files[kernel] = []
    features_files[kernel] = []

    for sycl_bench_file in os.listdir(sycl_bench_csv_dir):

        # comparison is done on file names: kernel is from the csv file, sycl_bench_file is the parsed log .csv
        if kernel == sycl_bench_file[::-1].split('_', 5)[5][::-1]: # we only take the kernel name from the sycl-bench result file
            features_files[kernel].append(f"{features_csv_dir}/{features_file}")
            # print(features_csv_dir+"/"+file)
            sycl_bench_files[kernel].append(f"{sycl_bench_csv_dir}/{sycl_bench_file}")
            # print(sycl_bench_csv_dir+"/"+file)

# print(sycl_bench_files)
# print(features_files)

df_all = pd.DataFrame()
for kernel in sycl_bench_files:
    df_kernel = pd.DataFrame()

    for sbench_file, features_file in zip(sycl_bench_files[kernel], features_files[kernel]):
        df_sbench = pd.read_csv(sbench_file)
        df_features = pd.read_csv(features_file)
        # remove feature kernel name
        df_features = df_features.drop(df_features.columns[0], axis=1)
        df_merged = pd.concat([df_sbench, df_features], axis=1)
        df_kernel = pd.concat([df_kernel, df_merged], ignore_index=True)

    df_all = pd.concat([df_all, df_kernel], ignore_index=True)
    if norm:
        core_freq=set(df_kernel["core-freq"])
        df_kernel["core-freq"] = df_kernel["core-freq"].apply(lambda x: (x - min(core_freq))/(max(core_freq)-min(core_freq)))

    df_kernel.to_csv(f"{merged_csv_dir}/merged_{kernel}.csv" , index=False, float_format='%.8f')

# df_all.to_csv(merged_csv_dir+"/merged_all.csv" , index=False, float_format='%.8f')

    # print(df_merged)
