#!/bin/bash
#PBS -N GeodSolv_SSV_250_300
#PBS -l nodes=cnode16:ppn=4
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
cd ..

source ~/.bashrc
python3 SamplingExperiment4.py --idx 250 300 >> SamplingExp_4_1_250_300.log
