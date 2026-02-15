#!/bin/bash

#SBATCH --job-name=MVP-collect-data
#SBATCH --partition=long
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#
#######################################


python3 sirs.py -t task3
