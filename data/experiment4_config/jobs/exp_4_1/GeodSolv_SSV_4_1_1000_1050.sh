#!/bin/bash
#PBS -N GeodSolv_SSV_1000_1050
#PBS -l nodes=cnode13:ppn=4
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
cd ..

source ~/.bashrc
python3 SamplingExperiment4.py --idx 1000 1050 >> SamplingExp_4_1_1000_1050.log
