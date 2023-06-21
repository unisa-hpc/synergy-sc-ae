#!/usr/bin/python3

import os, shutil
import sys
import pandas as pd
import getopt
import subprocess
import re


# extract default frequency on nvidia GPU
cmd_default_freq="nvidia-smi  -q | grep 'Default Applications Clocks' -A 2 | tail -n +2"

# Launch the command and capture the output
result = subprocess.check_output(cmd_default_freq, shell=True, stderr=subprocess.STDOUT)

# Convert the byte output to a string
output = result.decode("utf-8").strip()

# Find all numbers in the text using regular expressions
numbers = re.findall(r'\d+', output)
# Convert the matched numbers to integers
numbers = [int(number) for number in numbers]

defautl_core_freq=numbers[0]
default_memory_freq=numbers[1]

# take the path to the script folder
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
# path to source code with placeholder  
placeholder_dir = script_dir + "/cloverLeafApp/placeholder/src_cpy/"

# path to prediction folder
prediction_dir = script_dir + "/cloverLeafApp/placeholder/predicted_freq/"
# path to the source code for a specific target energy metric
min_edp_dir = script_dir + "/cloverLeafApp/placeholder/min_edp/"
min_ed2p_dir = script_dir + "/cloverLeafApp/placeholder/min_ed2p/"
default_dir = script_dir + "/cloverLeafApp/placeholder/default/"
es_50_dir = script_dir + "/cloverLeafApp/placeholder/es_50/"
pl_50_dir = script_dir + "/cloverLeafApp/placeholder/pl_50/"


paths_cpp_folder = [default_dir, min_edp_dir, min_ed2p_dir, es_50_dir, pl_50_dir]

for path in paths_cpp_folder:
    # create dir to strore the cpp files with the selected target frequency  
    os.makedirs(path, exist_ok=True)
    #remove all files from created dir
    os.system(f"rm -f {path}/*")


# contains the path to cpp file with final frequencies setted according to the specified target metric (e.g es_50, pl_50)
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

# for each cpp file we have to change the placeholder with the target energy metric
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
        with open(min_edp_dir+"/"+cpp_file, 'a') as min_edp_file:
            if("clover::execute($mem_freq, $core_freq, " in line):
                new_line = line.replace("clover::execute($mem_freq, $core_freq, ", "clover::execute(877, " + str(min_edp_freqs[i])+",")
           
            min_edp_file.write(new_line)  

        with open(min_ed2p_dir+"/"+cpp_file, 'a') as min_ed2p_file:
            if("clover::execute($mem_freq, $core_freq, " in line):
                new_line = line.replace("clover::execute($mem_freq, $core_freq, ", "clover::execute(877, " + str(min_ed2p_freqs[i])+",")
            min_ed2p_file.write(new_line)
        
        with open(es_50_dir+"/"+cpp_file, 'a') as es_50_file:
            if("clover::execute($mem_freq, $core_freq, " in line):
                new_line = line.replace("clover::execute($mem_freq, $core_freq, ", "clover::execute(877, " + str(es_50_freqs[i])+",")

            es_50_file.write(new_line)
        
        with open(pl_50_dir+"/"+cpp_file, 'a') as pl_50_file:
            if("clover::execute($mem_freq, $core_freq, " in line):
                new_line = line.replace("clover::execute($mem_freq, $core_freq, ", "clover::execute(877, " + str(pl_50_freqs[i])+",")

            pl_50_file.write(new_line)

        with open(default_dir+"/"+cpp_file, 'a') as default_file:
            if("clover::execute($mem_freq, $core_freq, " in line):
                new_line = line.replace("clover::execute($mem_freq, $core_freq, ", "clover::execute(" + str(default_memory_freq) +", " + str(defautl_core_freq) + ",")

            default_file.write(new_line)

        if("clover::execute($mem_freq, $core_freq, " in line):
            i = i+1


# parse cmd parameter to handle the cloverleaf compilation
argv = sys.argv[1:]

try:
    opts, argv = getopt.getopt(argv, "", ["cxx_compiler=","cxx_flags=", "cuda_support=","cuda_arch=", "rocm_support=", "rocm_arch="])
    #lets's check out how getopt parse the arguments
except:
    print('pass the arguments like -cxx_compiler= <path to cxx compiler> ')



cxx_compiler = ""
cxx_flags = ""
#in our experiment we use only the sycl version
sycl_runtime = "DPCPP"
# miniWeather requires pnetcdf
synergy_cuda_support="OFF"
synergy_cuda_arch=""
synergy_rocm_support="OFF"
synergy_rocm_arch=""

synergy_enable_profiling="ON"
synergy_sycl_backend="dpcpp"

for o,v in opts:
    if o in ['--cxx_compiler']:
        cxx_compiler = v
    elif o in ['--cxx_flags']:
        cxx_flags = v
    elif o in ['--cuda_support']:
        synergy_cuda_support = v   
    elif o in ['--cuda_arch']:
        synergy_cuda_arch = v   
    elif o in ['--rocm_support']:
        synergy_rocm_support = v   
    elif o in ['--rocm_arch']:
        synergy_rocm_arch = v   
         


# create the executables dir
os.makedirs(f"{script_dir}/executables", exist_ok=True)

# generate an executable for each target metric in paths_cpp_folder 
for folder in paths_cpp_folder:
    folder_name = os.path.basename(os.path.dirname(folder))
    for file in os.listdir(folder):
        # copy cpp file with frequency values setted in the src folder of the original application
        os.system(f"cp {folder}/{file} {script_dir}/cloverLeafApp/src")
        
    # compile and build cloverleaf    
    os.system(f"cmake -DCMAKE_CXX_COMPILER={cxx_compiler} \
                -DCMAKE_CXX_FLAGS=\"{cxx_flags}\" \
                -DSYCL_RUNTIME={sycl_runtime} \
                -DSYNERGY_CUDA_SUPPORT={synergy_cuda_support} \
                -DSYNERGY_CUDA_ARCH={synergy_cuda_arch} \
                -DSYNERGY_ROCM_SUPPORT={synergy_rocm_support} \
                -DSYNERGY_ROCM_ARCH={synergy_rocm_arch} \
                -DSYNERGY_SYCL_BACKEND={synergy_sycl_backend} \
                -S {script_dir}/cloverLeafApp/ -B {script_dir}/cloverLeafApp/build/")
    os.system(f"cmake --build {script_dir}/cloverLeafApp/build -j")
    # move executable in executables folder
    os.system(f"mv {script_dir}/cloverLeafApp/build/clover_leaf {script_dir}/executables/clover_leaf_{folder_name}")
    