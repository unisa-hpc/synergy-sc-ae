#!/usr/bin/python3

import os, shutil
import sys
import pandas as pd
import getopt
import subprocess
import re

# extract default frequency on nvidia GPU
cmd_default_freq="nvidia-smi  -q | grep 'Default Applications Clocks' -A 2 | tail -n +2"

result = subprocess.check_output(cmd_default_freq, shell=True, stderr=subprocess.STDOUT) # Launch the command and capture the output
output = result.decode("utf-8").strip() # Convert the byte output to a string
numbers = re.findall(r'\d+', output) # Find all numbers in the text using regular expressions
numbers = [int(number) for number in numbers] # Convert the matched numbers to integers

default_core_freq=numbers[0]
default_memory_freq=numbers[1]

# take the path to the script folder
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

placeholder_dir = script_dir + "/miniWeatherApp/placeholder/src_cpy/"

prediction_dir = script_dir + "/predictions/"

min_edp_dir = script_dir + "/miniWeatherApp/placeholder/min_edp/"
min_ed2p_dir = script_dir + "/miniWeatherApp/placeholder/min_ed2p/"
default_dir = script_dir + "/miniWeatherApp/placeholder/default/"
es_50_dir = script_dir + "/miniWeatherApp/placeholder/es_50/"
pl_50_dir = script_dir + "/miniWeatherApp/placeholder/pl_50/"

paths_cpp_folder = [default_dir, min_edp_dir, min_ed2p_dir, es_50_dir, pl_50_dir]

for path in paths_cpp_folder:
    # create dir to store the .cpp files with the selected target frequency  
    os.makedirs(path, exist_ok=True)
    # remove all files from created dir
    os.system(f"rm -f {path}/*")


# contains the path to cpp file with final frequencies set according to the specified target metric (e.g es_50, pl_50)
cpp_files = []
# contains the file with frequency values for each kernel in a cpp file
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
        with open(f"{default_dir}/{cpp_file}", 'a') as default_file:
            
            if("parallel_for($mem_freq, $core_freq, " in line):
                new_line = line.replace("parallel_for($mem_freq, $core_freq, ", f"parallel_for({default_memory_freq}, {default_core_freq},")
        
            default_file.write(new_line)
    
        with open(min_edp_dir+"/"+cpp_file, 'a') as min_edp_file:
            if("parallel_for($mem_freq, $core_freq, " in line):
               new_line = line.replace("parallel_for($mem_freq, $core_freq, ", f"parallel_for({default_memory_freq}, {min_edp_freqs[i]},")
            
            min_edp_file.write(new_line)  

        with open(min_ed2p_dir+"/"+cpp_file, 'a') as min_ed2p_file:
            if("parallel_for($mem_freq, $core_freq, " in line):
                new_line = line.replace("parallel_for($mem_freq, $core_freq, ", f"parallel_for({default_memory_freq}, {min_ed2p_freqs[i]},")
             
            min_ed2p_file.write(new_line)

        with open(es_50_dir+"/"+cpp_file, 'a') as es_50_file:
            if("parallel_for($mem_freq, $core_freq, " in line):
                new_line = line.replace("parallel_for($mem_freq, $core_freq, ", f"parallel_for({default_memory_freq}, {es_50_freqs[i]},")
        
            es_50_file.write(new_line)

        with open(pl_50_dir+"/"+cpp_file, 'a') as pl_50_file:
            if("parallel_for($mem_freq, $core_freq, " in line):
                new_line = line.replace("parallel_for($mem_freq, $core_freq, ", f"parallel_for({default_memory_freq}, {pl_50_freqs[i]},")
        
            pl_50_file.write(new_line)
        
        if("parallel_for($mem_freq, $core_freq, " in line):
            i=i+1


# problem input size for x and z dimension
nx_sizes = [3072, 4096, 5120, 6144, 7168]
nz_size = 1536
# parse the command line parameters for miniWeather compilation
argv = sys.argv[1:]
try:
    opts, argv = getopt.getopt(argv, "", ["cxx_compiler=","cxx_flags=", "sycl_flags=", "ldflags=", "cuda_arch="])
    #lets's check out how getopt parse the arguments
except:
    print('pass the arguments like --cxx_compiler= <path to cxx_compiler>')

cxx_compiler = ""
cxx_flags = ""
ldflags="-lpnetcdf" 
synergy_cuda_arch=""
yakl_sycl_flags=""

synergy_enable_profiling="ON"
synergy_sycl_backend="dpcpp"

for o,v in opts:
    if o in ['--cxx_compiler']:
        cxx_compiler = v
    elif o in ['--cxx_flags']:
        cxx_flags = v
    elif o in ['--sycl_flags']:
        yakl_sycl_flags = v    
    elif o in ['--cuda_arch']:
        synergy_cuda_arch = v

if cxx_compiler == "":
    print("Provide the absolute path to the DPC++ compiler as --cxx_compiler argument")          
    exit()

if synergy_cuda_arch == "":
    print("Provide the cuda architecture as --cuda_arch argument (e.g: sm_70)")          
    exit()

# create the executables dir
os.makedirs(f"{script_dir}/executables", exist_ok=True)

# compile miniWeather for each target metric and input. 
for folder in paths_cpp_folder:
    folder_name = os.path.basename(os.path.dirname(folder))

    for file in os.listdir(folder):
        # copy cpp file with frequency values setted in the src folder of the original application
        os.system(f"cp {folder}/{file} {script_dir}/miniWeatherApp/cpp/")
    
    # build the application for each input 
    i=1
    for nx_val in nx_sizes:
        os.system(f"cmake -DCMAKE_CXX_COMPILER={cxx_compiler} \
                    -DCMAKE_CXX_FLAGS=\"-fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3 {cxx_flags}\" \
                    -DYAKL_ARCH=SYCL \
                    -DYAKL_SYCL_FLAGS=\"-fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3 {yakl_sycl_flags}\" \
                    -DLDFLAGS={ldflags} \
                    -DSYNERGY_CUDA_SUPPORT=ON \
                    -DSYNERGY_CUDA_ARCH={synergy_cuda_arch} \
                    -DSYNERGY_SYCL_BACKEND={synergy_sycl_backend} \
                    -DNX={nx_val} \
                    -DNZ={nz_size} \
                    -S {script_dir}/miniWeatherApp/cpp -B {script_dir}/miniWeatherApp/cpp/build/")
        os.system(f"cmake --build {script_dir}/miniWeatherApp/cpp/build -j")
        # move the executable in the executables folder
        os.system(f"mv {script_dir}/miniWeatherApp/cpp/build/parallelfor {script_dir}/executables/parallel_for_{folder_name}_{i}")
        i=i*2
            
