#!/bin/bash                                                                         
#SBATCH --job-name=motion_depth                                                     
#SBATCH --ntasks=1                                                                  
#SBATCH --mem=12G                                                                    
#SBATCH --time=00:60:00                                                             
#SBATCH --mail-type=ALL                                                             
#SBATCH --output=script_run.out                                                
#SBATCH --gres=gpu:1 # Can be gpu:p100:1 or gpu:a100:1                              
#SBATCH --partition=normal                                                          
#SBATCH --qos=cpu                                                                   
                                                                                    
# Initialize conda for the Bash shell                                               
#python ./01_3d_approximation.py
#python ./02_nD_approximation.py
python ./03_3d_moment_matching.py


