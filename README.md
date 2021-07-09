# Video-Degradation
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
Detect *car* through Mask R-CNN for *night-street* video and store detection results in *Data/filtered/jackson-town-square/freq002_res64_mrcnn/*. Image resolution can be modified in the code file.

## 3. Error Bound Estimation
