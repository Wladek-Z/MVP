#!/bin/bash

#SBATCH --job-name=modify-data
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#
#######################################


python3 M_Trajectory.py
