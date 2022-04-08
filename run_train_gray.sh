#!/bin/bash

source ~/torch1.8/bin/activate

python train.py \
  --data ./data/MDSM7+smoking+ddd-small.yaml \
  --weights ./yolov5s.pt \
  --cfg ./models/yolov5s_gray.yaml \
  --hyp ./data/hyps/hyp.m7smoddd.yaml \
  --device 0 \
  --epochs 20 \
  --batch-size 4 \
  --workers 2 \
  --imgsz 640 \
  --name gray_cfg_test 2>&1 | tee -a gray_cfg_test5.log
  #--exist-ok \
  #--weights yolov5s.pt \
  # 2022.04.07

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

