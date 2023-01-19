# CLIP + Residual Feature Connection

Residual Feature Connection(RFC) aims to fuse the task-specific knowledge learned from the target domain and the original knowledge pre-trained from CLIP. We show that RFC can adapt pre-trained CLIP to downstream pathology tasks and achieve good performance with just a few annotated samples. Specifically, RFC achieves over 19% improvement on accuracy when only using 0.1% of labeled data in PCam with only 10 minutes of fine-tuning while running on a single RTX 2080Ti.

## Model Overview
<img width="750" alt="Residual_Feature_Connection_v5" src="https://user-images.githubusercontent.com/40489953/213370883-ed6b540b-de66-44f2-bc66-2d88d58b4f63.png">

## Installation
### Environment Set up
