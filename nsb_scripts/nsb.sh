#!/bin/sh


#SBATCH -p long

#SBATCH -J LSTnsb
                 
#SBATCH -N 1

ulimit -l unlimited
ulimit -s unlimited
ulimit -a

conda run -n  magic-lst python LSTnsb.py > nsblog.log 2>&1 
