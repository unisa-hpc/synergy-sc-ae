#!/usr/bin/python3
# cmd example
# python3 change_freq.py src_cpy/ predicted_freq/ max_perf/ min_energy/ min_edp/ min_ed2p/ default/
# aggiungere spostamento del file generato  in src, compilare e spostare l'eseguibile in executable 

import os, shutil
import sys
import pandas as pd

# per ogni file in src che termina con cpp scorrerere e ogni file in 
# prediction-csv
# scorrere il file .cpp fino a trovare execute($mem_freq, $core_freq
# e sotituire con le frequenza predette 877 e core_freq

# take the path to the script folder
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

placeholder_dir = script_dir + "/miniWeatherApp/placeholder/src_cpy/"

#file con le predizioni
prediction_dir = script_dir + "/miniWeatherApp/placeholder/predicted_freq/"
# cartelle di output con le varie configurazioni
min_edp_dir = script_dir + "/miniWeatherApp/placeholder/min_edp/"
min_ed2p_dir = script_dir + "/miniWeatherApp/placeholder/min_ed2p/"
default_dir = script_dir + "/miniWeatherApp/placeholder/default/"
es_50_dir = script_dir + "/miniWeatherApp/placeholder/es_50/"
pl_50_dir = script_dir + "/miniWeatherApp/placeholder/pl_50/"

paths_cpp_folder = [default_dir, min_edp_dir, min_ed2p_dir, es_50_dir, pl_50_dir]

for path in paths_cpp_folder:
    # create dir to strore the cpp files with the selected target frequency  
    os.makedirs(path, exist_ok=True)
    #remove all files from created dir
    os.system(f"rm -f {path}/*")


# contains the path to cpp file with final frequencies setted according to the specified target metric (e.g es_50, pl_50)
cpp_files = []
# contains tthe file with frequency values for each kernel in a cpp file
prediction_files = []

# for cloverleaf we have more cpp file  
for file in os.listdir(placeholder_dir):
    if(file.split(".")[1]=="cpp"):
        cpp_files.append(file)
cpp_files.sort()

# for each cpp file we have a prediction frequency file
for file in os.listdir(prediction_dir):
    prediction_files.append(file)

prediction_files.sort()


for cpp_file, prediction_file in zip(cpp_files, prediction_files):
    cpp_read_file = open(placeholder_dir+cpp_file, 'r')
    #creare 4 file nelle cartelle diverse 
    #sostituire le frequenze nella linea e salvare il file 
    lines_cpp_file = cpp_read_file.readlines()
    df_prediction = pd.read_csv(prediction_dir+prediction_file)

    min_edp_freqs = df_prediction["core_clk_edp"].values
    min_ed2p_freqs = df_prediction["core_clk_ed2p"].values
    es_50_freqs = df_prediction["clk_es_50"].values
    pl_50_freqs = df_prediction["clk_pl_50"].values
    
    if len(min_edp_freqs) == 0:
        continue
    
    i = 0
    for line in lines_cpp_file:
        new_line = line
        with open(default_dir+"/"+cpp_file, 'a') as default_file:
            
            if("parallel_for($mem_freq, $core_freq, " in line):
                new_line = line.replace("parallel_for($mem_freq, $core_freq, ", "parallel_for(877, 1312, ")
        
            default_file.write(new_line)
    
        with open(min_edp_dir+"/"+cpp_file, 'a') as min_edp_file:
            if("parallel_for($mem_freq, $core_freq, " in line):
               new_line = line.replace("parallel_for($mem_freq, $core_freq, ", "parallel_for(877,  " + str(min_edp_freqs[i])+",")
            
            min_edp_file.write(new_line)  

        with open(min_ed2p_dir+"/"+cpp_file, 'a') as min_ed2p_file:
            if("parallel_for($mem_freq, $core_freq, " in line):
                new_line = line.replace("parallel_for($mem_freq, $core_freq, ", "parallel_for(877,  " + str(min_ed2p_freqs[i])+",")
             
            min_ed2p_file.write(new_line)

        with open(es_50_dir+"/"+cpp_file, 'a') as es_50_file:
            if("parallel_for($mem_freq, $core_freq, " in line):
                new_line = line.replace("parallel_for($mem_freq, $core_freq, ", "parallel_for(877,  " + str(es_50_freqs[i])+",")
        
            es_50_file.write(new_line)

        with open(pl_50_dir+"/"+cpp_file, 'a') as pl_50_file:
            if("parallel_for($mem_freq, $core_freq, " in line):
                new_line = line.replace("parallel_for($mem_freq, $core_freq, ", "parallel_for(877,  " + str(pl_50_freqs[i])+",")
        
            pl_50_file.write(new_line)
        
        if("parallel_for($mem_freq, $core_freq, " in line):
            i=i+1

nx_sizes = [3072, 4096, 5120, 6144, 7168]
nz_size = 1536
for folder in paths_cpp_folder:
    folder_name = os.path.basename(os.path.dirname(folder))
    for file in os.listdir(folder):
        # copy cpp file with frequency values setted in the src folder of the original application
        os.system(f"cp {folder}/{file} {script_dir}/miniWeatherApp/cpp/")
        # build the application for each input
        i=1
        for nx_val in nx_sizes:
            os.system(f"cmake -DNX={nx_val} -DNZ={nz_size} -S {script_dir}/miniWeatherApp/cpp -B {script_dir}/miniWeatherApp/cpp/build/")
            os.system(f"cmake --build {script_dir}/miniWeatherApp/cpp/build -j")
            os.system(f"mv {script_dir}/miniWeatherApp/cpp/build/parallel_for {script_dir}/executables/parallel_for_{folder_name}_{i}")
            i=i*2
            
#cmake -DNX=3072 -DNZ=1536  ..

#             cmake --build . -j
# mv parallelfor  ../../../miniWeather-executable-versions/parallel_for_${1}_1

# for each dir containing the cpp with the target frequency do a cpy in the src folder and the compile the program for each target input 
# then move th executable in the executables folder
