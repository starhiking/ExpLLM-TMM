# 融合scrfd 标签的 box 框 和 原始的人脸框
import os
import json
import numpy as np
import cv2

scrfd_json = "datasets/tools/scrfd_results/UTK_face_wild_results.json"

utkface_path = "/mnt/data/lanxing/UTK-face-wild/images"

merge_bbox_file_path = "datasets/tools/scrfd_results/list_bbox_utkface_new.txt"

scrfd_json_data = json.load(open(scrfd_json, "r"))

flag_head = False
with open(merge_bbox_file_path, "w") as f, open(os.path.join('datasets/tools/anno_json', "utkface_box.json"), "w") as f_box, open(os.path.join('datasets/tools/anno_json', "utkface_all.json"), "w") as f_all:
    f.write(f"{len(scrfd_json_data)}\n")
    f.write("image_id x_1 y_1 width height\n")

    for scrfd_data in scrfd_json_data:
        img_path = scrfd_data["path"]

        assert os.path.exists(os.path.join(utkface_path, img_path)), f"{img_path} not exists"

        if len(img_path.split("_")) != 4:
            print(f"{img_path} not in the right format, filter")
            continue

        if len(scrfd_data['result']) == 0:
            print(f"{img_path} no face detected, filter")
            # img = cv2.imread(os.path.join(utkface_path, img_path))
            # f.write(f"{img_path} 0 0 {img.shape[1]} {img.shape[0]}")
            continue
        
        if len(scrfd_data['result']) > 1:
            print(f"{img_path} more than one face detected, filter")
            continue
        
        print(f"processing {img_path}")
        age_gender_race = img_path.split("_")[:3]
        age = int(age_gender_race[0]) # 0-116
        gender = int(age_gender_race[1]) # 0 for male, 1 for female
        race = int(age_gender_race[2]) # 0 for white, 1 for black, 2 for asian, 3 for indian, 4 for others
        face_box = [int(float(x)) for x in scrfd_data['result'][0]['face_1']['facial_area']]

        # convert age: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+ to 0,1,2,....,8
        age = 0 + int(age>=3) + int(age>=10) + int(age>=20) + int(age>=30) + int(age>=40) + int(age>=50) + int(age>=60) + int(age>=70)

        # convert gender
        gender = int(gender == 0) # 1 for male, 0 for female

        f.write(f"{img_path} {face_box[0]} {face_box[1]} {face_box[2]} {face_box[3]}\n")

        if not flag_head:
            f_box.write("[")
            f_all.write("[")
            flag_head = True
        else:
            f_box.write(",\n")
            f_all.write(",\n")
        
        box_dict = {
            "file_name": img_path,
            "bbox": face_box
        }

        all_dict = box_dict.copy()
        all_dict.update({
            "age_gender_race": [age,gender,race]
        })

        f_box.write(json.dumps(box_dict))
        f_all.write(json.dumps(all_dict))
    
    f_box.write("]")
    f_all.write("]")