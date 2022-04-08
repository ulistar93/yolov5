#!/bin/bash

source ~/torch1.8/bin/activate
#export QT_LOGGING_RULES='*.debug=false;*.debug=false;qt.qpa.xcb.xcberror.warning=false;qt.qpa.xcb.xcberror.error=false;qt.qpa.xcb.warning=false;qt.qpa.xcb.error=false;qt.qpa.xcb=false'
#export DISPLAY=:0.0
python detect.py \
  --weight runs/train/MDSM7+smoking+ddd/weights/best.pt \
  --source /home/ycm/d/yolov5_runs/20220404_163733_DRO.mp4 \
  --data data/MDSM7+smoking+ddd.yaml \
  --imgsz 320 \
  --device 0 \
  --conf-thres 0.6 \
  --iou-thres 0.3 \
  --name MDSM7+smoking+ddd/tt/20220404_163733_DRO-320.mp4

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
