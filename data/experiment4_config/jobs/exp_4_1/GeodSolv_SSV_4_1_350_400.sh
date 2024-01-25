#!/bin/bash
#PBS -N GeodSolv_SSV_350_400
#PBS -l nodes=cnode12:ppn=4
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
cd ..

source ~/.bashrc
python3 SamplingExperiment4.py --idx 350 400 >> SamplingExp_4_1_350_400.log
