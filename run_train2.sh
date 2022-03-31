#!/bin/bash

source ~/torch1.8/bin/activate

python train.py \
  --data data/coco2017phone.yaml \
  --weights ./runs/train/phone_h100_l20_aug_nolrf/weights/last.pt \
  --hyp ./data/hyps/hyp.scratch-low-nolrf-phone.yaml \
  --device 0 \
  --epochs 20 \
  --batch-size 4 \
  --name phone_h100_l20_aug_nolrf \
  --exist-ok
  #--resume ./runs/train/phone_h100_l20_aug_nolrf/weights/last.pt \ # not work

  # 2022.03.30
  #--data data/coco2017phone.yaml \
  #--weights yolov5x.pt \
  #--hyp ./data/hyps/hyp.scratch-high-nolrf.yaml \
  #--device 0 \
  #--epochs 100 \
  #--batch-size 4 \
  #--name phone_hiaug_nolrf \


  #--name phone_h100_l20_aug_nolrf \
  #--hyp ./data/hyps/hyp.scratch-low-nolrf.yaml \
  #--data data/coco128.yaml \
  #--weights yolov5x.pt \
  #--workers 2 \
