## ObjectDetection_Yolov3

-----------------------------------

### You can RUN YoloDetectorBot by following to next steps:
#### 1) Download Repository and `yolov3(tiny).h5` and put model into `models/yolov3(tiny).h5`.
##### Link to the models .h5: https://www.kaggle.com/kanametov/yolov3
#### 2) RUN `bot.py` using Powershell and command `python bot.py`
#### 3) or RUN `Bot.ipynb`
#### 4) Finaly open Telegram Bot: https://web.telegram.org/@YoloDetectorBot

---------------------------------------

### Before to RUN model :

##### 1) Download `yolov3.h5` and `yolov3(tiny).h5` models and put them into the folder `models/` as `models/yolov3.h5` and `models/yolov3(tiny).h5`.
##### * You can download just `yolov3(tiny).h5` and run all on **Tiny version of YOLOv3**. 
##### Link to the models .h5: https://www.kaggle.com/kanametov/yolov3
##### 2** If you are going to train your own model on VOC Dataset download the dataset using the link below and put it into the folder `dataset/` as follows `dataset/VOC2007`.
##### Link to the VOC Dataset: https://www.kaggle.com/kanametov/vocdataset

--------------------------------------------

### RUN model :

#### To RUN Object Detector on *picture*:

##### 1) With Notebook: use PictureDetection.ipynb
##### 2) With Windows Powershell: use command: `python detect_picture.py`

#### Parameters of `detect_picture.py`

| Parameters       |   Short Commands  |    Full Commands   |  Default value   |
|------------------|-------------------|--------------------|------------------|
|   Image          |   `-i`            | `--input`          |    `dog.jpg`     |
|Image size        |   `-is`           |`--input_size`      |    `416`         |
|Score Threshold   |   `-st`           |`--score_threshold` |    `0.25`        |
|IoUnion Threshold |   `-it`           |`--iou_threshold`   |    `0.5`         |

------------------------------------------

#### To RUN Object Detector on *web-camera*:

##### 1) With Notebook: use WebDetection.ipynb
##### 2) With Windows Powershell: use command: `python detect_web.py`

#### Parameters of `detect_web.py`

| Parameters       |   Short Commands  |    Full Commands   |  Default value   |
|------------------|-------------------|--------------------|------------------|
|Image size        |   `-is`           |`--input_size`      |    `256`         |
|Score Threshold   |   `-st`           |`--score_threshold` |    `0.25`        |
|IoUnion Threshold |   `-it`           |`--iou_threshold`   |    `0.5`         |

##### Press `q` to exit from Web-Detector.
