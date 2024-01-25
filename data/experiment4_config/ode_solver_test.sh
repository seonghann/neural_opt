#!/bin/bash
#PBS -N GeodSolv_SSV_0_100
#PBS -l nodes=cnode11:ppn=4
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
cd ..

source ~/.bashrc
python3 SamplingExperiment4.py --idx 0 2
