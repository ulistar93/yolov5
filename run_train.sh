#!/bin/bash

source ~/torch1.8/bin/activate

python train.py \
  --data data/coco128.yaml \
  --weights yolov5x.pt \
  --device 0 \
  --epochs 10 \
  --batch-size 4 \
  #--workers 2 \
