#!/bin/bash
#PBS -N test_exp_4_1
#PBS -l nodes=cnode10:ppn=4
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
cd ..

source ~/.bashrc
python3 SamplingExperiment4.py --idx 0 2 >> test.log
