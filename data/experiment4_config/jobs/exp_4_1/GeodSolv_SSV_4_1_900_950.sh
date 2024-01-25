#!/bin/bash
#PBS -N GeodSolv_SSV_900_950
#PBS -l nodes=cnode11:ppn=4
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
cd ..

source ~/.bashrc
python3 SamplingExperiment4.py --idx 900 950 >> SamplingExp_4_1_900_950.log
