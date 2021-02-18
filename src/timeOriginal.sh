#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 1
#SBATCH --mem-per-cpu=3000
#SBATCH --output=timeOriginal.txt
#SBATCH -w gnode65


cd /home/shanmukh.alle/parkinsonsfromgait/src

python timeOriginal.py

