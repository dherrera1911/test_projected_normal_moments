#!/bin/bash
#SBATCH --job-name=prnorm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=script_run.out
#SBATCH --partition=normal
#SBATCH --qos=normal

# Initialize conda for the Bash shell
#python -u 01_moments_approximation.py par_approx_3d.yaml
#python -u 01_moments_approximation.py par_approx_nd.yaml
#python -u 02_moment_matching.py par_mm_3d.yaml
#python -u 02_moment_matching.py par_mm_nd.yaml

