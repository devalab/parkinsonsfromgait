#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 1
#SBATCH --mem-per-cpu=3000
#SBATCH --output=timeLPresidual.txt
#SBATCH -w gnode65


cd /home/shanmukh.alle/parkinsonsfromgait/src

matlab -batch "generateLPresidual; exit"

