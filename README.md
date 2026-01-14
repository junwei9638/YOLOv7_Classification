# YOLOv7 Angle Classification for Vehicle Detection in Aerial Images

This repository implements an angle classification module using the YOLOv7 backbone, specifically designed for **Vehicle Detection in Aerial Images**. It predicts the orientation of vehicles to facilitate the conversion of Horizontal Bounding Boxes (HBB) to Oriented Bounding Boxes (OBB).

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Post-Processing](#post-processing)
- [Pre-trained Models](#pre-trained-models)

## Overview
This project provides a pipeline for training and deploying an angle classification model tailored for **Vehicle Detection in Aerial Images**. It includes scripts for dataset preparation, model training, inference, and visualization of oriented bounding boxes for aerial imagery.

## Requirements
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Dataset Preparation
### Step 1: Format Your Dataset
Ensure your dataset is in the Horizontal Bounding Box (HBB) format with angle information.
**Format:** `angle(0-359), x, y, w, h`

## Usage

### Training
**Step 2: Train the Angle Classification Model**
Use the `train.py` script to train the model. You can customize the training parameters as needed.

**Example Command:**
```bash
python classify/train.py \
  --data data/rotate.yaml \
  --epochs 40 \
  --img 224 \
  --cfg models/yolov7_backbone_cspElan.yaml \
  --hyp data/hyps/hyp_rotate.yaml \
  --csl 5 \
  --name <your_experiment_name> \
  --workers 6 \
  --batch-size 32 \
  --optimizer AdamW \
  --device 0 \
  --thresh 5
```

### Inference
**Step 3: Run Model Inference**
Run the `predict.py` script to perform inference using your trained weights.

**Example Command:**
```bash
python classify/predict.py \
  --weights <path_to_model_weights> \
  --source <path_to_images_or_txt> \
  --name <your_experiment_name> \
  --thresh 5 \
  --data data/rotate.yaml
```

## Post-Processing
**Step 4: Generate Rotated Images**
Due to limitations in direct OBB calculation, this step rotates original images by 45 degrees to help generate accurate rotation labels using an object detection model.

```bash
python classify/create_45_img.py --path <path_to_images>
```

**Step 5: Visualize Results**
Draw the final oriented bounding boxes based on the inference results.

```bash
python classify/choosebox_and_draw.py \
  --name <your_experiment_name> \
  --ori_img <path_to_original_images> \
  --pred_label <label_from_step3> \
  --rlabel <label_from_rotated_inference>
```

## Pre-trained Models
You can download the pre-trained angle classification model here:
- [angle_cls.pt](https://github.com/junwei9638/YOLOv7_Classification/blob/0b643dff766a03f2714dcc3541a75a525de49486/anlge_cls.pt)
