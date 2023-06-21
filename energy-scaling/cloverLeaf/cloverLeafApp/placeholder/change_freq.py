#!/usr/bin/python3
import os, shutil
import sys
import pandas as pd

# per ogni file in src che termina con cpp scorrerere e ogni file in 
# prediction-csv
# scorrere il file .cpp fino a trovare execute($mem_freq, $core_freq
# e sotituire con le frequenza predette 877 e core_freq

if len(sys.argv) < 7:
    print("Insert path to sycl-bench folder as command line argument")
    exit(0)


cpp_dir = sys.argv[1]
prediction_dir = sys.argv[2]
max_perf_dir = sys.argv[3]
min_energy_dir = sys.argv[4]
min_edp_dir = sys.argv[5]
min_ed2p_dir = sys.argv[6] 


shutil.rmtree(max_perf_dir)
shutil.rmtree(min_energy_dir)
shutil.rmtree(min_edp_dir)
shutil.rmtree(min_ed2p_dir)

os.mkdir(max_perf_dir)
os.mkdir(min_energy_dir)
os.mkdir(min_edp_dir)
os.mkdir(min_ed2p_dir)

cpp_files = []
prediction_files = []

for file in os.listdir(cpp_dir):
    if(file.split(".")[1]=="cpp"):
        cpp_files.append(file)
    
cpp_files.sort()

for file in os.listdir(prediction_dir):
    prediction_files.append(file)

prediction_files.sort()



for cpp_file, prediction_file in zip(cpp_files, prediction_files):
    cpp_read_file = open(cpp_dir+cpp_file, 'r')
    #creare 4 file nelle cartelle diverse 
    #sostituire le frequenze nella linea e salvare il file 
    lines_cpp_file = cpp_read_file.readlines()
    df_prediction = pd.read_csv(prediction_dir+prediction_file)

    max_perf_freqs = df_prediction["core_clk_time"].values
    min_energy_freqs = df_prediction["core_clk_energy"].values
    min_edp_freqs = df_prediction["core_clk_edp"].values
    min_ed2p_freqs = df_prediction["core_clk_ed2p"].values
   

    if len(max_perf_freqs) == 0:
        continue
    
    i = 0
    for line in lines_cpp_file:
        with open(max_perf_dir+"/"+cpp_file, 'a') as max_perf_file:
            if("clover::execute($mem_freq, $core_freq, " in line):
                max_perf_file.write(line.replace("clover::execute($mem_freq, $core_freq, ", "clover::execute(877, " + str(max_perf_freqs[i])+","))
                i=i+1
            else: 
                max_perf_file.write(line)  

    i = 0
    for line in lines_cpp_file:
        with open(min_energy_dir+"/"+cpp_file, 'a') as min_energy_file:
            if("clover::execute($mem_freq, $core_freq, " in line):
                min_energy_file.write(line.replace("clover::execute($mem_freq, $core_freq, ", "clover::execute(877, " + str(min_energy_freqs[i])+","))
                i=i+1
            else: 
                min_energy_file.write(line)  
    i = 0
    for line in lines_cpp_file:
        with open(min_edp_dir+"/"+cpp_file, 'a') as min_edp_file:
            if("clover::execute($mem_freq, $core_freq, " in line):
                min_edp_file.write(line.replace("clover::execute($mem_freq, $core_freq, ", "clover::execute(877, " + str(min_edp_freqs[i])+","))
                i=i+1
            else: 
                min_edp_file.write(line)  

    
    i = 0
    for line in lines_cpp_file:
        with open(min_ed2p_dir+"/"+cpp_file, 'a') as min_ed2p_file:
            if("clover::execute($mem_freq, $core_freq, " in line):
                min_ed2p_file.write(line.replace("clover::execute($mem_freq, $core_freq, ", "clover::execute(877, " + str(min_ed2p_freqs[i])+","))
                i=i+1
            else: 
                min_ed2p_file.write(line)