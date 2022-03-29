#!/bin/bash

source ~/torch1.8/bin/activate

python train.py \
  --data data/coco2017phone.yaml \
  --weights runs/train/phone_only/weights/last.pt \
  --hyp ./data/hyps/hyp.scratch-low-phone.yaml \
  --device 0 \
  --epochs 20 \
  --batch-size 4 \
  --name phone_only
  #--data data/coco128.yaml \
  #--weights yolov5x.pt \
  #--workers 2 \
