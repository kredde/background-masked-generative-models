#!/bin/bash
#SBATCH --partition mllab  # the partition for the Lab course
#SBATCH -N 1 # number of nodes
#SBATCH --gres=gpu:1 # number of GPUs to be allocated
#SBATCH -t 1-23:00 # time after which the process will be killed (D-HH:MM)
#SBATCH -o "/nfs/homedirs/%u/slurm-output/slurm-%j.out"  # where the output log will be stored
#SBATCH --mem=16000 # the memory (MB) that is allocated to the job. If your job exceeds this it will be killed -- but don't set it too large since it will block resources and will lead to your job being given a low priority by the scheduler.
#SBATCH --qos=interactivelab   # this line ensures a very high priority (e.g. start a Jupyter notebook) but only one job per user can run under this mode (remove for normal compute jobs).
 
cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

tensorboard --logdir ./lightning_logs --bind_all &

# Activate your conda environment if necessary
# conda activate myenv
export XDG_RUNTIME_DIR="" # Fixes Jupyter bug with read/write permissions https://github.com/jupyter/notebook/issues/1318
jupyter notebook --no-browser --ip=$(hostname).kdd.in.tum.de
