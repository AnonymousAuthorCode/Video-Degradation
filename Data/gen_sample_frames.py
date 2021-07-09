import argparse
import os

import cv2
import swag
import tqdm
import numpy as np
from generate_fnames import get_video_fname

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./')
    parser.add_argument('--out_dir', default='./sample_frames/')
    parser.add_argument('--base_name', required=True)
    parser.add_argument('--date', default='2017-12-17')
    args = parser.parse_args()

    DATA_PATH = args.data_path
    base_name = args.base_name
    date = args.date
    OUT_DIR = args.out_dir

    if base_name == "jackson-town-square":
        OUT_DIR = os.path.join(OUT_DIR, base_name, "freq002_resori")

        video_fname = get_video_fname(DATA_PATH, base_name, date, load_video=True)

        cap = swag.VideoCapture(video_fname)
        nb_frames = cap.cum_frames[-1]

        for i in tqdm.tqdm(range(nb_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            if(i%50 == 0):
                cv2.imwrite(os.path.join(OUT_DIR,"%d.jpg"%i),frame)

    elif base_name == "UA-DETRAC":
        OUT_DIR = os.path.join(OUT_DIR, base_name, "freq1_resori")

        video_fname = "./svideo/UA-DETRAC/Insight-MVT_Annotation_Test/"
        folders = ["MVI_39031", "MVI_39051", "MVI_39211", "MVI_39271", "MVI_39311", "MVI_39361", "MVI_39371", "MVI_39401", "MVI_39501", "MVI_39511", "MVI_40701", "MVI_40711"]

        frame_count = 0
        for folder in folders:
            IN_DIR = video_fname + folder + '/'
            files = os.listdir(IN_DIR)
            for filename in files:
                file_dir = IN_DIR + filename
                frame = cv2.imread(file_dir)
                cv2.imwrite(os.path.join(OUT_DIR,"%d.jpg"%frame_count),frame)
                frame_count += 1
    
    else:
        print("Please reenter video name.")

if __name__ == '__main__':
    main()
