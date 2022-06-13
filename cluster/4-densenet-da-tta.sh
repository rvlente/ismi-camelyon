#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
module load 2021
module load TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1

# Copy project to scratch
cp -r "$HOME/ismi-camelyon" "$TMPDIR"
cd "$TMPDIR/ismi-camelyon"

# Install requirements.
pip install -r requirements.txt

# Run scripts.
python train.py "densenet" --da 2
python test.py "submission.csv" "densenet" \
    "$HOME/patch_camelyon/camelyonpatch_level_2_split_test_x.h5" \
    "$HOME/patch_camelyon/camelyonpatch_level_2_split_test_y.h5" --tta

# Copy output to results folder.
RESULTS_FOLDER="$HOME/ismi-camelyon/results/4-densenet-da-tta/$(date '+%Y%m%dT%H%M%S')"
mkdir -p $RESULTS_FOLDER
cp *.csv $RESULTS_FOLDER
cp *.json $RESULTS_FOLDER
cp *.hdf5 $RESULTS_FOLDER
cp -r logs $RESULTS_FOLDER
