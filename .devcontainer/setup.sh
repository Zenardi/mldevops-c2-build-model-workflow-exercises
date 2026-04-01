#!/bin/zsh
set -e

echo "==> Creating udacity conda environment..."
conda create --name udacity python=3.8 mlflow jupyter pandas matplotlib requests -c conda-forge -y

echo "==> Installing root pip dependencies..."
pip install -r requirements.txt

echo "==> Logging in to Weights & Biases..."
if [ -n "${WANDB_API_KEY}" ]; then
    wandb login "${WANDB_API_KEY}"
    echo "    wandb login successful."
else
    echo "    WANDB_API_KEY not set — skipping wandb login."
fi
