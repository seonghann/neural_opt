#!/bin/bash
#PBS -N GeodSolv_SSV_200_250
#PBS -l nodes=cnode15:ppn=4
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
cd ..

source ~/.bashrc
python3 SamplingExperiment4.py --idx 200 250 >> SamplingExp_4_1_200_250.log
