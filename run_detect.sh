#!/bin/bash

source ~/torch1.7/bin/activate
#export QT_LOGGING_RULES='*.debug=false;*.debug=false;qt.qpa.xcb.xcberror.warning=false;qt.qpa.xcb.xcberror.error=false;qt.qpa.xcb.warning=false;qt.qpa.xcb.error=false;qt.qpa.xcb=false'
#export DISPLAY=:0.0
python detect.py \
  --weight yolov5x6.pt \
  --source ~/z/Vision/Datasets/smokingVSnotsmoking/validation_data/smoking/ \
  --view-img \
  --save-txt \
  --save-conf \
  --classes 67

  #--device cpu \
  #--source ~/z/Vision/Datasets/smokingVSnotsmoking/training_data/smoking/*.jpg \
  #--source ~/z/Vision/Datasets/smokingVSnotsmoking/training_data/notsmoking/*.jpg \
  #--source ~/z/Vision/Datasets/smokingVSnotsmoking/validation_data/smoking/*.jpg \
  #--source ~/z/Vision/Datasets/smokingVSnotsmoking/validation_data/notsmoking/*.jpg \
  #--source ~/z/Vision/Datasets/smokingVSnotsmoking/testing_data/*.jpg \
