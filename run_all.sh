#!/bin/bash

# Exit immediately if a command fails
set -e

# Get the absolute path of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"  # Assuming this script is in SRW/

echo "=== Running from repo root: $REPO_ROOT ==="

# Move to repo root
#cd "$REPO_ROOT"

echo "=== Setting up Conda environment ==="
# Create environment if it doesn't exist
CONDA_BASE=$(conda info --base) 
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda env create -f ./environment.yml || echo "Environment 'srw' already exists"
conda activate srw

echo "=== Downloading pretrained checkpoints ==="
mkdir -p results

git clone https://huggingface.co/vithurabimans/SRW ./SRW/results || echo "Repo already exists"
echo "=== Running training ==="



MODE="$1"

if [[ "$MODE" == "train" ]]; then
    echo "=== Running training ==="
    echo "You can visualize training logs with TensorBoard:"
    echo "tensorboard --logdir $REPO_ROOT/results --port 6006"
    echo "Then open http://localhost:6006 in your browser."
    
    python -m SRW.train --dataset_config ./SRW/configs/dataset.yaml --exp_config ./SRW/configs/train.yaml

    echo "=== Training done! Check outputs in results/ ==="

elif [[ "$MODE" == "test" ]]; then
    echo "=== Running evaluation/testing ==="
    python -m SRW.test --dataset_config ./SRW/configs/dataset.yaml --exp_config ./SRW/configs/test.yaml
    echo "=== Testing done! ==="

else
    echo "Usage: $0 [train|test]"
    exit 1
fi
