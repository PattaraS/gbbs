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
  for file_idx, filename in enumerate(files):
    for mult_factor in mult_factors:
        for iteration in iterations:
            for nw in num_workers:
                for option in options:
                    out_path_components = ["ParallelDensestSubgraph", filename,
                            mult_factor, iteration, nw, option, ".out"]
                    out_filename = os.path.join(write_dir, "_".join(out_path_components))
                    ss = ("PARLAY_NUM_THREADS=" + str(nw) + " ./DensestSubgraph -s -m " + str(compressed[0]) + " " + "-rounds " + str(rounds[0]) + " -iter " + iteration + " -option " + option + " -approx_kcore_base " + mult_factor + " " + read_dir + filename)
                    print(ss)
                    out = shellGetOutput(ss)
                    appendToFile(out, out_filename)

if __name__ == "__main__":
  main()
