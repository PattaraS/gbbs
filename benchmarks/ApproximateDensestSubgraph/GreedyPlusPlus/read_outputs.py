import os
import sys
import signal
import time
import subprocess

def signal_handler(signal,frame):
  print("bye")
  sys.exit(0)
signal.signal(signal.SIGINT,signal_handler)

def shellGetOutput(str) :
  process = subprocess.Popen(str,shell=True,stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
  output, err = process.communicate()

  if (len(err) > 0):
    raise NameError(str+"\n"+output.decode('utf-8')+err.decode('utf-8'))
  return output.decode('utf-8')

def appendToFile(out, filename):
  with open(filename, "a+") as out_file:
    out_file.writelines(out)

def optionToType(option):
    if option == 0:
        return "Approx k core"
    if option == 1:
        return "Exact k core"
    if option == 2:
        return "Ceiling of density"
    if option == 3:
        return "Max k core over 2"
    if option == 4:
        return "No preprocessing"

def main():
  # Read parameters from setup file
  with open('dsg_setup.txt') as parameters_file:
    for line in parameters_file:
      line = line.strip()
      split = [x.strip() for x in line.split(':')]
      if len(split) <= 1:
        continue
      params = [x.strip() for x in split[1].split(',')]
      if line.startswith("Input graph directory"):
        read_dir = split[1]
      elif line.startswith("Graphs"):
        files = params.copy()
      elif line.startswith("Output directory"):
        write_dir = split[1]
      elif line.startswith("Numbers of workers"):
        num_workers = params.copy()
      elif line.startswith("Approx Mult Factor"):
        mult_factors = params.copy()
      elif line.startswith("Rounds"):
        rounds = params.copy()
      elif line.startswith("Iterations"):
        iterations = params.copy()
      elif line.startswith("Options"):
        options = params.copy()
      elif line.startswith("Compressed"):
        compressed = params.copy()

  # Setup other parameters
  program_dir = "../benchmarks/"
  empty = "../benchmarks/EdgeOrientation/ParallelLDS/empty_h"
  for file_idx, filename in enumerate(files):
    for mult_factor in mult_factors:
        for iteration in iterations:
            for nw in num_workers:
                for option in options:
                    out_path_components = ["ParallelDensestSubgraph", filename,
                            mult_factor, iteration, nw, option, ".out"]
                    read_filename = os.path.join(write_dir, "_".join(out_path_components))

                    best_time = 10000
                    best_density = 0

                    with open(read_filename, "r") as read_file:
                        for line in read_file:
                            line = line.strip()
                            split = [x.strip() for x in line.split(':')]
                            if split[0].startswith("### Total core time"):
                                if float(split[1]) < best_time:
                                    best_time = float(split[1])
                            if split[0].startswith("### Density of current Densest Subgraph is"):
                                if float(split[1]) > best_density:
                                    best_density = float(split[1])

                    print(str(filename), end = ",")
                    print(str(iteration), end = ",")
                    print(str(optionToType(int(option))), end = ",")
                    print(str(nw), end = ",")
                    print(str(best_time), end = ",")
                    print(str(best_density), end = "\n")

if __name__ == "__main__":
  main()
