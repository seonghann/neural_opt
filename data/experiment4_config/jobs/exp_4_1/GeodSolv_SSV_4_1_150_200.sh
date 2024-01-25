#!/bin/bash
#PBS -N GeodSolv_SSV_150_200
#PBS -l nodes=cnode14:ppn=4
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
cd ..

source ~/.bashrc
python3 SamplingExperiment4.py --idx 150 200 >> SamplingExp_4_1_150_200.log
