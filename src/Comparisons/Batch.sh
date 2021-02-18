#!/bin/bash
#SBATCH -n 1
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=3000
#SBATCH --time=4-00:00:00
#SBATCH --output=timeMaachietal.txt
#SBATCH -w gnode65

cd /home/shanmukh.alle/parkinsonsfromgait/src/Comparisons
python TimeMaachietal.py