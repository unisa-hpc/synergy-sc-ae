import os
import sys

if len(sys.argv) != 2:
    print("Insert the name of the data folder")
    exit(0)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

logs_dir = f"{script_dir}/{sys.argv[1]}"
output_dir = f"{script_dir}/parsed"

if "provided-logs" in logs_dir:
  replicate = True 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

configurations = ["default", "min_ed2p", "min_edp", "es_50", "pl_50"]

for conf in configurations:

  with open(f"{output_dir}/miniweather_{conf}_ws.csv", "w") as ws_out:
    ws_out.write("size,ngpus,time,energy\n")

    for scaling in ["ws"]:
      out_file = ws_out

      for nodes in [1, 2, 4, 8, 16]:
        ngpus = nodes * 4
        time_mean = 0
        energy_mean = 0

        out_file.write(str(32*ngpus)+",")
        out_file.write(str(ngpus)+",")

        logs = os.listdir(logs_dir)
        num_logs = max(set(map(lambda log: int(log.split(".log")[0][-1]), filter(lambda log_file: conf in log_file, logs)))) if not replicate else 2

        size = 0
        for i in range(1, num_logs + 1):
          with open(f"{logs_dir}/miniweather_{conf}_{scaling}_{nodes}_{i}.log", "r") as input_file:
            energy = 0.0
            time = 0.0
            for line in input_file:
              if "Node name" in line:
                energy += float(line.split(" ")[9])
              if "CPU Time" in line:
                time = float(line.split(" ")[2])
              if "global_nx" in line:
                size = int(line.split(" ")[1].replace(",", ""))
            
            energy_mean+=energy
            time_mean+=time
          
        out_file.write(str(size)+",")
        out_file.write(str(ngpus)+",")
        out_file.write(str(round(time_mean/num_logs, 5))+",")
        out_file.write(str(round(energy_mean/num_logs, 5))+"\n")