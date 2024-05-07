#!/bin/bash



# Execute the Python script with provided parameters
python3 kinetic-san.py \
  --b1 0.5 \
  --b2 0.999 \
  --batch_size 128 \
  --channels 2 \
  --checkpoint_interval 100 \
  --lambda_gp 10 \
  --dataset h36m \
  --latent_dim 512 \
  --mlp_dim 8 \
  --lr 0.0002 \
  --n_classes 9 \
  --n_cpu 8 \
  --n_critic 5 \
  --n_epochs 4000 \
  --sample_interval 100 \
  --t_size 1 \
  --v_size 16 \
  --csv_path "../data/kp_16_cover_modes/cover2/traincover2.csv" \
  --runs "runs_cover2_test" \
  --tb_runs "tensorboard_runs_cover2_test" \
  --device 0 \
  --model "san"