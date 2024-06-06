#!/bin/bash
#SBATCH --job-name=prnorm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=script_run.out
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort

# Initialize conda for the Bash shell
#python 01_moments_approximation.py par_approx_3d.yaml
#python 01_moments_approximation.py par_approx_nd.yaml
#python 02_moment_matching.py par_approx_3d.yaml
#python 02_moment_matching.py par_approx_nd.yaml

