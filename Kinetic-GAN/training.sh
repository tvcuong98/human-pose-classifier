#!/bin/bash

# Default values
b1=0.5
b2=0.999
batch_size=32
channels=2
checkpoint_interval=1000
d_interval=1
dataset='h36m'
latent_dim=512
lr=0.0002
n_classes=10
n_cpu=8
n_critic=5
n_epochs=1000
sample_interval=1000
t_size=64
v_size=16
csv_path="/ske/data/gt16.csv"


# Execute the Python script with provided parameters
python3 kinetic-gan.py \
  --b1 0.5 \
  --b2 0.999 \
  --batch_size 128 \
  --channels 2 \
  --checkpoint_interval 40527 \
  --dataset h36m \
  --latent_dim 512 \
  --mlp_dim 8 \
  --lr 0.0002 \
  --n_classes 9 \
  --n_cpu 8 \
  --n_critic 5 \
  --n_epochs 4000 \
  --sample_interval 40527 \
  --t_size 32 \
  --v_size 16 \
  --csv_path "../data/kp_16_cover_modes/mixed/trainmixed.csv" \
  --runs "runs_mixed" \
  --tb_runs "tensorboard_runs_mixed"