#!/bin/bash
#PBS -N GeodSolv_SSV_850_900
#PBS -l nodes=cnode16:ppn=4
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
cd ..

source ~/.bashrc
python3 SamplingExperiment4.py --idx 850 900 >> SamplingExp_4_1_850_900.log
