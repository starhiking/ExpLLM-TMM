import os
import json
import pandas as pd

emotion_list = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
output_folder_path = 'data_list/anno_json/'

# first process 7cls and 8cls txt

AffectNet_280k_path = '/mnt/data/lanxing/AffectNet_processed/'
files = os.listdir(AffectNet_280k_path)
files = [file for file in files if file.endswith('.txt')]
assert len(files) == 4

for file in files:
    file_path = os.path.join(AffectNet_280k_path, file)
    output_json_path = os.path.join(output_folder_path, "Affectnet_280k_" + file.replace('.txt', '.json'))
    with open(file_path, 'r') as f:
        lines = f.readlines()
    f.close()
    output_json = []
    train_or_val_folder_name = 'trainnew' if 'train' in file else 'validnew'
    for line in lines:
        line = line.strip().split(' ')
        image_name = os.path.join(train_or_val_folder_name, line[0]) # 添加相对路径
        assert os.path.exists(os.path.join(AffectNet_280k_path, image_name))
        emotion_label = int(line[1])
        emotion = emotion_list[emotion_label]
        output_json.append({'file_name': image_name, 'emo': emotion})

    with open(output_json_path, 'w') as f:
        json.dump(output_json, f, indent=4)
    f.close()


# then process the affectnet_kaggle csv
AffectNet_Kaggle_path = '/mnt/data/lanxing/AffectNet-kaggle/'

files = os.listdir(AffectNet_Kaggle_path)
files = [file for file in files if file.endswith('.csv')]
assert len(files) == 2

for file in files:
    file_path = os.path.join(AffectNet_Kaggle_path, file)
    output_json_path = os.path.join(output_folder_path, "Affectnet_kaggle_" + file.replace('.csv', '.json'))
    df = pd.read_csv(file_path)
    output_json = []
    for index, row in df.iterrows():
        image_name = row['image'].replace('../input/affectnetsample/','')
        assert os.path.exists(os.path.join(AffectNet_Kaggle_path, image_name))
        emotion_label = int(float(row['emotion'])) - 1 # csv 要减 1
        emotion = emotion_list[emotion_label]
        output_json.append({'file_name': image_name, 'emo': emotion})

    with open(output_json_path, 'w') as f:
        json.dump(output_json, f, indent=4)
    f.close()


