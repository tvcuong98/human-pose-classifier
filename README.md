Implementing an effective Graph Convolutional Neural Network for Classification of Human Pose

# Preparation
### Install torchlight
Run `pip install torch `

### Preprocessing datasets.

#### Dataset MPII was used

2D Human Pose Estimation: New Benchmark and State of the Art Analysis

### Data Processing
+ Create a folder that contain all the images 
+ Use human_pose.py for YOLOV8 human pose estimation
+ Reformat the data from 17 keypoint to 16 keypoint


### Training


+ torch_heavy_pose_classifier.py for training data using normal fully connected layer
+ torch_robust_pose_classifier.py for training using GCN pose classifier
+ ray_torch_pose_classifier.py using ray_tune for tuning




## Acknowledgements

This repo is based on [Kinetic-GAN](https://github.com/DegardinBruno/Kinetic-GAN)

Thanks to the original authors for their work!