#!/bin/bash

source ~/torch1.8/bin/activate

python train.py \
  --data ./data/MDSM7+smoking+ddd.yaml \
  --weights '' \
  --hyp ./data/hyps/hyp.m7smoddd.yaml \
  --cfg ./models/yolov5s_gray.yaml \
  --device 0 \
  --epochs 100 \
  --batch-size 4 \
  --name MDSM7+smoking+ddd-gray_s
  #--workers 4 \
  # 2022.04.11

  #--data ./data/MDSM7+smoking+ddd.yaml \
  #--weights yolov5l.pt \
  #--hyp ./data/hyps/hyp.m7smoddd.yaml \
  #--device 0 \
  #--epochs 100 \
  #--batch-size 16 \
  #--workers 2 \
  #--name MDSM7+smoking+ddd-lowaug
  # 2022.04.07

  #--data ./data/MDSM7+smoking+ddd.yaml \
  #--weights yolov5l.pt \
  #--hyp ./data/hyps/hyp.scratch-high.yaml \
  #--device 0 \
  #--epochs 100 \
  #--batch-size 16 \
  #--workers 2 \
  #--name MDSM7+smoking+ddd
  # 2022.04.06

  #--data ./data/SmokeFiltered.yaml \
  #--weights yolov5l.pt \
  #--hyp ./data/hyps/hyp.scratch-high.yaml \
  #--device 0 \
  #--epochs 100 \
  #--batch-size 16 \
  #--workers 2 \
  #--name SmokeFiltered16
  # 2022.03.31

  # 2022.03.29
  #--data data/coco2017phone.yaml \
  #--weights runs/train/phone_only/weights/last.pt \
  #--hyp ./data/hyps/hyp.scratch-low-phone.yaml \
  #--device 0 \
  #--epochs 20 \
  #--batch-size 4 \
  #--name phone_only

