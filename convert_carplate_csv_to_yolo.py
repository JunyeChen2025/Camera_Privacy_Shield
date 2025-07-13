# convert_carplate_csv_to_yolo.py

import os
import pandas as pd

def convert_annotations(csv_path, img_folder, label_folder):
    os.makedirs(label_folder, exist_ok=True)
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        img_w = row['width']
        img_h = row['height']

        x_center = ((row['xmin'] + row['xmax']) / 2) / img_w
        y_center = ((row['ymin'] + row['ymax']) / 2) / img_h
        width = (row['xmax'] - row['xmin']) / img_w
        height = (row['ymax'] - row['ymin']) / img_h

        label_file = os.path.join(label_folder, os.path.splitext(row['filename'])[0] + '.txt')
        with open(label_file, 'a') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


base_path = "Dataset/Carplate"

splits = ['train', 'valid', 'test']

for split in splits:
    csv_file = os.path.join(base_path, split, '_annotations.csv')
    img_folder = os.path.join(base_path, split, 'images')
    label_folder = os.path.join(base_path, split, 'labels')
    convert_annotations(csv_file, img_folder, label_folder)
    print(f"✅ Finished converting: {csv_file} → {label_folder}")
