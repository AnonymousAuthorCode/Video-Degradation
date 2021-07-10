# Video-Degradation
## Requirements
- wget
- Python 3.7
- opencv-python 4.1.2
- facenet-pytorch
- Pillow 6.1.0
- scikit-image
- pandas

## 1. Video Data Preparation
Download *night-street* video in the folder *Data/svideo/jackson-town-square/* and unzip it
```
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1KhDzedVoiiWVD_pIJl4IGEeTi-QkO927' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KhDzedVoiiWVD_pIJl4IGEeTi-QkO927" -O 2017-12-17.zip && rm -rf /tmp/cookies.txt
```
Download *UA-DETRAC* video in the folder *Data/svideo/UA-DETRAC/* and unzip it
```
wget http://detrac-db.rit.albany.edu/Data/DETRAC-test-data.zip
```
Sample and store frames in the folders *Data/sample_frames/jackson-town-square/freq002_resori/* and *Data/sample_frames/UA-DETRAC/freq1_resori/*
```
cd Data
python gen_sample_frames.py --base_name jackson-town-square
python gen_sample_frames.py --base_name UA-DETRAC
```

## 2. Frame Object Detection
Detect *face* through MTCNN and store detection results in *Data/filtered/jackson-town-square/freq002_face_mtcnn/* and *Data/filtered/UA-DETRAC/freq1_face_mtcnn/*
```
cd Detectors
python face_detection.py --base_name jackson-town-square
python face_detection.py --base_name UA-DETRAC
```
Detect *car* through Mask R-CNN for *night-street* video and store detection results in *Data/filtered/jackson-town-square/freq002_res64_mrcnn/*. Clone and install [Mask_RCNN](https://github.com/matterport/Mask_RCNN) in the folder *Detectors*, and copy *Detectors/mask_rcnn.py* to *Detectors/Mask_RCNN/*. Image resolution can be modified in *mask_rcnn.py*
```
cd Detectors/Mask_RCNN
python mask_rcnn.py
```
Detect *car* and *person* through YOLOv4 for *UA-DETRAC* video and store detection results in *Data/filtered/UA-DETRAC/freq1_res64_yolo/*. Clone and install [darknet](https://github.com/AlexeyAB/darknet) in the folder *Detectors*, and copy *Detectors/darknet.py* to *Detectors/darknet/*. Image resolution can be modified in *Detectors/darknet/cfg/yolov4.cfg*
```
cd Detectors/darknet
python darknet.py
```

## 3. Error Bound Estimation
Compute the true relative error and error bound. Take the following case (aggregate function = AVG, video = night-street, reduced resolution = 64\*64, frame sampling fraction = 0.01, restricted class = face, correction set size fraction = 0.5, method = ours) as an example:
```
cd Estimation
python run_answer_quality_avg.py --obj_name car --base_name jackson-town-square --test_date 2017-12-17 --ground_truth freq002_res640_mrcnn --ground_truth_freq 0.02 --predictions freq002_res64_mrcnn --sample_frac 0.01 --constraints freq002_face_mtcnn --cons_obj_name face --use_val True --val_frac 0.5 --our_alg True
```
The commands are similar for other aggregate functions.
