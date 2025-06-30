import os
import json
import random
# FairFace 按照标准分，带 train 的是训练集，带 val 的是验证集
# UTKFace 按比例随机分，train 0.8, val 0.2

############ 处理 FairFace
with open('datasets/tools/anno_json/fairface_box.json', 'r') as f_box, open('datasets/tools/anno_json/fairface_all.json', 'r') as f_all:
    box_json = json.load(f_box)
    all_json = json.load(f_all)

f_box.close()
f_all.close()

train_box_json = []
valid_box_json = []

train_all_json = []
valid_all_json = []

for box_data, all_data in zip(box_json, all_json):
    assert box_data['file_name'] == all_data['file_name']
    file_name = box_data['file_name']

    train_or_val = file_name.split('/')[0]

    if train_or_val == 'train':
        train_box_json.append(box_data)
        train_all_json.append(all_data)
    # elif file_name in valid_list:
    elif train_or_val == 'val':
        valid_box_json.append(box_data)
        valid_all_json.append(all_data)
    else:
        raise ValueError('Unknown file name: {}'.format(file_name))

assert len(train_box_json) + len(valid_box_json)  == len(box_json)
assert len(train_all_json) + len(valid_all_json)  == len(all_json)

print('FairFace Train, Val :', len(train_box_json), len(valid_box_json))


with open('datasets/tools/anno_json/fairface_box_train.json', 'w') as f_box, open('datasets/tools/anno_json/fairface_all_train.json', 'w') as f_all:
    json.dump(train_box_json, f_box)
    json.dump(train_all_json, f_all)

f_box.close()
f_all.close()

with open('datasets/tools/anno_json/fairface_box_valid.json', 'w') as f_box, open('datasets/tools/anno_json/fairface_all_valid.json', 'w') as f_all:
    json.dump(valid_box_json, f_box)
    json.dump(valid_all_json, f_all)


f_box.close()
f_all.close()


########### 处理 UTKFace
with open('datasets/tools/anno_json/utkface_box.json', 'r') as f_box, open('datasets/tools/anno_json/utkface_all.json', 'r') as f_all:
    box_json = json.load(f_box)
    all_json = json.load(f_all)

assert len(box_json) == len(all_json)

train_index = random.sample(range(len(box_json)), int(len(box_json) * 0.8)) # 80% train
valid_index = list(set(range(len(box_json))) - set(train_index)) # 20% valid
train_index.sort()
valid_index.sort()

print('UTKFace Train, Val :', len(train_index), len(valid_index))

# check
cat_index = train_index + valid_index
cat_index.sort()
assert cat_index == list(range(len(box_json))), 'train_index + valid_index != list(range(len(box_json)))'

train_box_json = [box_json[train_i] for train_i in train_index]
valid_box_json = [box_json[valid_i] for valid_i in valid_index]

train_all_json = [all_json[train_i] for train_i in train_index]
valid_all_json = [all_json[valid_i] for valid_i in valid_index]

assert len(train_box_json) + len(valid_box_json)  == len(box_json)
assert len(train_all_json) + len(valid_all_json)  == len(all_json)

with open('datasets/tools/anno_json/utkface_box_train.json', 'w') as f_box, open('datasets/tools/anno_json/utkface_all_train.json', 'w') as f_all:
    json.dump(train_box_json, f_box)
    json.dump(train_all_json, f_all)

f_box.close()
f_all.close()

with open('datasets/tools/anno_json/utkface_box_valid.json', 'w') as f_box, open('datasets/tools/anno_json/utkface_all_valid.json', 'w') as f_all:
    json.dump(valid_box_json, f_box)
    json.dump(valid_all_json, f_all)

f_box.close()
f_all.close()
