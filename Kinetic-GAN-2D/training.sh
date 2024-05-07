#!/bin/bash



# Execute the Python script with provided parameters
python3 kinetic-gan.py \
  --b1 0.5 \
  --b2 0.999 \
  --batch_size 512 \
  --channels 2 \
  --checkpoint_interval 500 \
  --dataset h36m \
  --latent_dim 512 \
  --mlp_dim 8 \
  --lr 0.0002 \
  --n_classes 9 \
  --n_cpu 8 \
  --n_critic 5 \
  --n_epochs 30000 \
  --sample_interval 500 \
  --t_size 1 \
  --v_size 16 \
  --csv_path "../data/kp_16_cover_modes/mixed/trainmixed.csv" \
  --runs "runs_mixed" \
  --tb_runs "tensorboard_runs_mixed"