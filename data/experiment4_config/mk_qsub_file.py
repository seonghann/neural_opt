# make qsub files for each experiment
#
#
"""
#!/bin/bash
#PBS -N GeodSolv_SSV_0_100
#PBS -l nodes=cnode11:ppn=4
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
cd ..

source ~/.bashrc
python3 SamplingExperiment4.py --idx 100 200 >> SamplingExp_4_1_100_200.log
"""

# we will make 24 qsub files with 24 different idx values
# 0-th file : idx 0-50
# 1-th file : idx 50-100 ... (increment 50)
# and we have cnode11 ~ cnode16, each has 16 cores
# set proper file name and node name

for i in range(24):
    file_name = "GeodSolv_SSV_4_1_" + str(i*50) + "_" + str((i+1)*50) + ".sh"
    node_name = "cnode" + str(i%6 + 11)
    with open(file_name, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#PBS -N GeodSolv_SSV_" + str(i*50) + "_" + str((i+1)*50) + "\n")
        f.write("#PBS -l nodes=" + node_name + ":ppn=4\n")
        f.write("#PBS -l walltime=500:00:00\n\n")
        f.write("cd $PBS_O_WORKDIR\n")
        f.write("cd ..\n\n")
        f.write("source ~/.bashrc\n")
        f.write("python3 SamplingExperiment4.py --idx " + str(i*50) + " " + str((i+1)*50) + " >> SamplingExp_4_1_" + str(i*50) + "_" + str((i+1)*50) + ".log\n")
