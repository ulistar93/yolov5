#!/bin/bash

source ~/torch1.8/bin/activate

for isz in "320" "640"
do
  #echo python train.py \
  python train.py \
    --data ./data/MDSM7+smoking+ddd.yaml \
    --weights ./yolov5n.pt \
    --hyp ./data/hyps/hyp.m7smoddd.yaml \
    --cfg ./models/yolov5n.yaml \
    --imgsz ${isz} \
    --device 0 \
    --batch-size 16 \
    --workers 2 \
    --epochs 300 \
    --name m7smoddd-rgb_${isz} 2>&1 | tee -a m7smoddd-rgb_${isz}.log
done
  # 2022.04.25

  #--data ./data/MDSM7+smoking+ddd.yaml \
  #--weights '' \
  #--hyp ./data/hyps/hyp.m7smoddd.yaml \
  #--cfg ./models/yolov5n_gray.yaml \
  #--device 0 \
  #--imgsz 320 \
  #--batch-size 16 \
  #--workers 2 \
  #--epochs 300 \
  #--name MDSM7+smoking+ddd-gray_n_w8
  #--cfg ./models/yolov5n.yaml \
  #--name MDSM7+smoking+ddd-rgb_n_nopt
  # 2022.04.15
  
  #--data ./data/MDSM7+smoking+ddd.yaml \
  #--weights '' \
  #--hyp ./data/hyps/hyp.m7smoddd.yaml \
  #--cfg ./models/yolov5s_gray.yaml \
  #--device 0 \
  #--epochs 100 \
  #--batch-size 4 \
  #--name MDSM7+smoking+ddd-gray_s
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

