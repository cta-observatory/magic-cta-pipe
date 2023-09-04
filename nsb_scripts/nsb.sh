#!/bin/sh


#SBATCH -p long

#SBATCH -J LSTnsb
                 
#SBATCH -N 1

ulimit -l unlimited
ulimit -s unlimited
ulimit -a

start_time=`date +%s`
conda run -n  magic-lst python LSTnsb.py > nsblog.log 2>&1
end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.

