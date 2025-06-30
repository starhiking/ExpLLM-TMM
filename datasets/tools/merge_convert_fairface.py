# 融合scrfd 标签的 box 框 和 原始的人脸框
import os
import json
import numpy as np
import cv2
import pandas as pd

scrfd_json = "datasets/tools/scrfd_results/FairFace_1.25_results.json"

fairface_path = "/mnt/data/lanxing/fairface-1.25"

train_csv_path = os.path.join(fairface_path, "fairface_label_train.csv")
val_csv_path = os.path.join(fairface_path, "fairface_label_val.csv")

merge_bbox_file_path = "datasets/tools/scrfd_results/list_bbox_fairface_new.txt"

scrfd_json_data = json.load(open(scrfd_json, "r"))


# check 数据容量是否一致， 另外调整 scrfd_json_data 的排序方式
train_scrfd_json_data = []
val_scrfd_json_data = []
for data in scrfd_json_data:
    if data["path"].startswith("train"):
        train_scrfd_json_data.append(data)
    elif data["path"].startswith("val"):
        val_scrfd_json_data.append(data)
    else:
        print(f"{data['path']} not in train or val")
        break

# sort
train_scrfd_json_data = sorted(train_scrfd_json_data, key=lambda x: int(x["path"].split("/")[1][:-4]))
val_scrfd_json_data = sorted(val_scrfd_json_data, key=lambda x: int(x["path"].split("/")[1][:-4]))

scrfd_json_data = train_scrfd_json_data + val_scrfd_json_data

# for json_line in train_scrfd_json_data[-10:] + val_scrfd_json_data[-10:]:
#     print(json_line["path"])
    

# load csv
train_csv = pd.read_csv(train_csv_path)
val_csv = pd.read_csv(val_csv_path)
csv_data = pd.concat([train_csv, val_csv])

# check number 
print(f"train_csv: {len(train_csv)}, val_csv: {len(val_csv)}")
print(f"train_scrfd_json_data: {len(train_scrfd_json_data)}, val_scrfd_json_data: {len(val_scrfd_json_data)}")
assert len(train_csv) == len(train_scrfd_json_data), f"{len(train_csv)} != {len(train_scrfd_json_data)}"
assert len(val_csv) == len(val_scrfd_json_data), f"{len(val_csv)} != {len(val_scrfd_json_data)}"

print(f"csv_data: {len(csv_data)}, scrfd_json_data: {len(scrfd_json_data)}")

flag_head = False
with open(merge_bbox_file_path, "w") as f, open(os.path.join('datasets/tools/anno_json', "fairface_box.json"), "w") as f_box, open(os.path.join('datasets/tools/anno_json', "fairface_all.json"), "w") as f_all:
    f.write(f"{len(scrfd_json_data)}\n")
    f.write("image_id x_1 y_1 width height\n")

    for csv_line, scrfd_data in zip(csv_data.iterrows(), scrfd_json_data):
        # print(csv_line[0]) # index
        # print(csv_line[1]) # data
        csv_line = csv_line[1]
        img_path = scrfd_data["path"]
        assert csv_line["file"] == img_path, f"{csv_line['file']} != {img_path}"

        age = csv_line["age"]
        gender = csv_line["gender"]
        race = csv_line["race"]

        assert os.path.exists(os.path.join(fairface_path, img_path)), f"{img_path} not exists"

        if len(scrfd_data['result']) == 0:
            print(f"{img_path} no face detected, filter")
            # img = cv2.imread(os.path.join(fairface_path, img_path))
            # f.write(f"{img_path} 0 0 {img.shape[1]} {img.shape[0]}")
            continue
        
        if len(scrfd_data['result']) > 1: # 有多个人脸， 标签无法对齐， 原标签没有提供人脸框，不能对应具体人脸
            print(f"{img_path} more than one face detected, filter")
            continue
        
        print(f"processing {img_path}, convert age, gender, race")

        age_dict = {
            '0-2': 0,
            '3-9': 1,
            '10-19': 2,
            '20-29': 3,
            '30-39': 4,
            '40-49': 5,
            '50-59': 6,
            '60-69': 7,
            'more than 70': 8,
        }

        # 1 for Male, 0 for Female
        gender_dict = {
            'Male': 1,
            'Female': 0
        }

        # 0 for white, 1 for black, 2 for asian, 3 for indian, 4 for others
        race_dict = {
            'White': 0,
            'Black': 1,
            'East Asian': 2,
            'Southeast Asian': 2,
            'Indian': 3,
            'Latino_Hispanic':4,
            'Middle Eastern': 4, 
        }

        # print(f"age: {age}, gender: {gender}, race: {race}")
        age = age_dict[age]
        gender = gender_dict[gender]
        race = race_dict[race]
        # print(f"age: {age}, gender: {gender}, race: {race}")

        face_box = [int(float(x)) for x in scrfd_data['result'][0]['face_1']['facial_area']]

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