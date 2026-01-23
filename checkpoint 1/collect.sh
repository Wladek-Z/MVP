#!/bin/bash

#SBATCH --job-name=MVP-collect-data
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#
#######################################


python3 task456_collect.py
