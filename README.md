# ObjectDetection_Yolov3


## Before to RUN the model :

#### 1) Download `yolov3.h5` and `yolov3(tiny).h5` models and put them into the folder `models/` as `models/yolov3.h5` and `models/yolov3(tiny).h5`.
#### Link to the models .h5: https://www.kaggle.com/kanametov/yolov3
#### 2** If you are going to train your own model on VOC Dataset download the dataset using the link below and put it into the folder `dataset/` as follows `dataset/VOC2007`.
#### Link to the VOC Dataset: https://www.kaggle.com/kanametov/vocdataset

## RUN the model :

### To RUN Object Detector on *picture*:

#### 1) With Notebook: use PictureDetection.ipynb
#### 2) With Windows Powershell: use command: `python detect_picture.py --input=dog.jpg --input_size=416` (default values for `input=dog.jpg` and `input_size=416`)

### To RUN Object Detector on *web-camera*:

#### 1) With Notebook: use WebDetection.ipynb
#### 2) With Windows Powershell: use command: `python detect_web.py --input_size=256` (default value for `input_size=256`)
