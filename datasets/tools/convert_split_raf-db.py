import os
import json

# 没有用 scrfd 进行人脸检测框

# rafdb 由于使用的是已经裁剪的数据，所以只有文件名和表情标签

patition_label_path = '/mnt/data/lanxing/RAF-DB/basic/EmoLabel/list_patition_label.txt'

with open(patition_label_path, 'r') as f:
    lines = f.readlines()
f.close()

# each row: <image_name> <emotion label>

emotion_list = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

train_json_list = []
valid_json_list = []

for line in lines:
    line = line.strip()
    image_name, emotion_label = line.split(' ')
    emotion_label = int(emotion_label) - 1 # 1: surprise, 2: fear, 3: disgust, 4: happiness, 5: sadness, 6: anger, 7: neutral
    emotion = emotion_list[emotion_label]
    aligned_img_name = image_name.replace('.jpg', '_aligned.jpg')
    if 'train' in aligned_img_name:
        train_json_list.append({'file_name': aligned_img_name, 'emo': emotion})
    elif 'test' in aligned_img_name:
        valid_json_list.append({'file_name': aligned_img_name, 'emo': emotion})
    else:
        raise ValueError('Invalid image name')

with open('data_list/anno_json/rafdb_train.json', 'w') as f:
    json.dump(train_json_list, f, indent=4)
f.close()

with open('data_list/anno_json/rafdb_valid.json', 'w') as f:
    json.dump(valid_json_list, f, indent=4)
f.close()