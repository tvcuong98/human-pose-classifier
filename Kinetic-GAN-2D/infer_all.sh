#!/bin/bash

# Default values
# Execute the Python script with provided parameters
python3 infer_all.py \
  --model_type "RobustPoseClassifier" \
  --classimodel "/ske/classifier/output/robust_pose_classifier/mixed/good_checkpoints/best_ckpt_ep349_iterloss0.001513181270898453_valloss0.07580063150574763_valacc0.9853508095605242.pt" \
  --mode "hard" \
  --generator "/ske/Kinetic-GAN/runs-small/kinetic-gan/models/generator_300900.pth" \
  --anglefilter False \
  --maxsample 5000 \
  --batchsize 1024 \
  --num_vis 20 \
  --output gan_sample_hard