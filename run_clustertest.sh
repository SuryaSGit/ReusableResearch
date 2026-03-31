#!/bin/bash
#
# --------------------------------------------------------------
#  Batch script for running clustertest.py on the Campus Cluster
# --------------------------------------------------------------

# ------------------- SLURM resource requests -------------------
#SBATCH --job-name=clustertest
#SBATCH --output=clustertest.out
#SBATCH --error=clustertest.err
#SBATCH --time=01:00:00
#SBATCH --account=sridhar-ic
#SBATCH --partition=ic-express
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:h100.1g.20gb:1
#SBATCH --mem=8G

# ------------------- Load the programming environment ----------
module load python/3.13.2


# ------------------- Run the Python program --------------------
python clustertest.py