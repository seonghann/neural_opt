#!/bin/bash
#PBS -N GeodSolv_SSV_450_500
#PBS -l nodes=cnode14:ppn=4
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
cd ..

source ~/.bashrc
python3 SamplingExperiment4.py --idx 450 500 >> SamplingExp_4_1_450_500.log
