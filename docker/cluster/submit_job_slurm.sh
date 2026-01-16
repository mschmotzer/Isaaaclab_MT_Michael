#!/usr/bin/env bash
module load eth_proxy

cat <<EOT > job.sh
#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=40
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=6000
#SBATCH --tmp=120g
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=mschmotzer@ethz.ch
#SBATCH --job-name="Test_aug$(date +"%Y-%m-%dT%H:%M")"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT

sbatch < job.sh
rm job.sh

# Copy all results from node-local scratch to project storage
# mkdir -p /cluster/project/meboldt/lucas/job_results/
# rsync -av "$TMPDIR/euler_test_2k.hdf5" /cluster/project/meboldt/lucas/job_results/