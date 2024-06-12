#!/bin/bash

# Add email, change QOS to regular if necessary, change job name in -J

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J data_conversion
#SBATCH --mail-user=[EMAIL]
#SBATCH --mail-type=ALL
#SBATCH -A m3623
#SBATCH -t 0:30:0

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load conda
conda activate /global/common/software/m3623/exaepi

# Output file should not exist prior to script running
OUTPUT_DIR="data.hdf5"
DATA_DIR="/dvs_ro/cfs/projectdirs/m3623/test/bay/"
# For larger sets (e.g. USA data), stripe_large may be better
stripe_medium $OUTPUT_DIR
srun -n 32 -c 8 --cpu_bind=cores python convert_hdf5.py $DATA_DIR $OUTPUT_DIR
