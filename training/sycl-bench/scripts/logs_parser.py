#!/usr/bin/python3

import os
import sys

if len(sys.argv) != 3:
    print("Insert path to sycl-bench folder as command line argument")
    exit(0)

work_dir=sys.argv[1]
out_dir=sys.argv[2]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for file in os.listdir(work_dir):
    out_file = file.replace(".log","")
    with open(work_dir+"/"+file, "r") as input_file, open(out_dir+"/"+out_file+"_parsed.csv", "w") as output_file:
        output_file.write("kernel-name,size,num-iters,core-freq,memory-freq,kernel-time [s],run-time [s],mean-energy [J],max-energy [J]\n")
        for line in input_file:
            if "Results for" in line:
                line = line.replace("*", "")
                line = line.replace("Results for", "")
                line = line.replace(" ", "")
                line = line.replace("\n", "")
                output_file.write(line)
            if "core-freq:" in line:
                line = line.replace("core-freq:", "")
                line = line.replace(" ", "")
                line = line.replace("\n", "")
                output_file.write(", "+ line)
            if "memory-freq:" in line:
                line = line.replace("memory-freq:", "")
                line = line.replace(" ", "")
                line = line.replace("\n", "")
                output_file.write(", "+ line)  
            if "problem-size:" in line:
                line = line.replace("problem-size:", "")
                line = line.replace(" ", "")
                line = line.replace("\n", "")
                output_file.write(", "+ line)  
            if "num-iters:" in line:
                line = line.replace("num-iters:", "")
                line = line.replace(" ", "")
                line = line.replace("\n", "")
                output_file.write(", "+ line)
            if "kernel-time-mean:" in line:
                line = line.replace("kernel-time-mean:", "")
                line = line.replace("[s]", "")  
                line = line.replace(" ", "")
                line = line.replace("\n", "")
                output_file.write(", "+ line)
            if "run-time-mean:" in line:
                line = line.replace("run-time-mean:", "")
                line = line.replace("[s]", "")  
                line = line.replace(" ", "")
                line = line.replace("\n", "")
                output_file.write(", "+ line)
            if "kernel-energy-mean:" in line:
                line = line.replace("kernel-energy-mean:", "")
                line = line.replace("[J]", "")  
                line = line.replace(" ", "")
                line = line.replace("\n", "")
                output_file.write(", "+ line)
            if "kernel-energy-max:" in line:
                line = line.replace("kernel-energy-max:", "")
                line = line.replace("[J]", "")  
                line = line.replace(" ", "")
                output_file.write(", "+ line)
            
            
                    

        