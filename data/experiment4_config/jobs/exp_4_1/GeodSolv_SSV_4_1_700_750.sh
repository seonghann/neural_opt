#!/bin/bash
#PBS -N GeodSolv_SSV_700_750
#PBS -l nodes=cnode13:ppn=4
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
cd ..

source ~/.bashrc
python3 SamplingExperiment4.py --idx 700 750 >> SamplingExp_4_1_700_750.log
