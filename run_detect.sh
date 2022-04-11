#!/bin/bash

source ~/torch1.8/bin/activate
#export QT_LOGGING_RULES='*.debug=false;*.debug=false;qt.qpa.xcb.xcberror.warning=false;qt.qpa.xcb.xcberror.error=false;qt.qpa.xcb.warning=false;qt.qpa.xcb.error=false;qt.qpa.xcb=false'
#export DISPLAY=:0.0
python detect.py \
  --weight runs/train/MDSM7+smoking+ddd/weights/best.pt \
  --source /home/ycm/eval_videos/ \
  --data data/MDSM7+smoking+ddd.yaml \
  --device 0 \
  --conf-thres 0.6 \
  --iou-thres 0.3 \
  --name m7smoddd \
  # 2022.04.11
  #--weight runs/train/MDSM7+smoking+ddd-lowaug/weights/best.pt \
  #--name m7smoddd-lowaug \
  #--exist-ok 

  #--weight runs/train/SmokeFiltered16/weights/best.pt \
  #--source /home/ycm/z/Vision/Datasets/MDSM/eval_dataSet/videos/ \
  #--data data/SmokeFiltered.yaml \
  #--device 0 \
  #--conf-thres 0.6 \
  #--iou-thres 0.3 \
  #--name SmokeFiltered16 \
  # 2022.04.06
  
  #--weight runs/train/exp9/weights/best.pt \
  #--source /home/ycm/z/Vision/Datasets/Smoking_data_filtered/train+val/train+val_images/ \
  #--data data/coco2017pcb.yaml \
  #--conf-thres 0.6 \
  #--iou-thres 0.3 \
  #--view-img \
  #--save-txt \
  #--save-conf \

  #--weight runs/train/exp9/weights/best.pt \ #cell phone, cup, bottle

  #--source ~/z/Vision/Datasets/Smoking_data_filtered/testing_data/ \
  #--source ~/z/Vision/Datasets/Dataset\ containing\ smoking\ and\ not-smoking\ images/Dataset\ containing\ smoking\ and\ not-smoking\ images\ \(smoker\ vs\ non-smoker\)/smokingVSnotsmoking/dataset/validation_data/smoking/ \
#--source smokingVSnotsmoking/validation_data/smoking/ \
  #--device cpu \
  #--source ~/z/Vision/Datasets/smokingVSnotsmoking/training_data/smoking/*.jpg \
  #--source ~/z/Vision/Datasets/smokingVSnotsmoking/training_data/notsmoking/*.jpg \
  #--source ~/z/Vision/Datasets/smokingVSnotsmoking/validation_data/smoking/*.jpg \
  #--source ~/z/Vision/Datasets/smokingVSnotsmoking/validation_data/notsmoking/*.jpg \
  #--source ~/z/Vision/Datasets/smokingVSnotsmoking/testing_data/*.jpg \
