#!/bin/bash

#SBATCH --job-name=mvp-task4
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#
#######################################


python3 task4-5-6.py
