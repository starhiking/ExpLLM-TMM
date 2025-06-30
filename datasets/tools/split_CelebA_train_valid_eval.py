import os
import json

with open('datasets/tools/split_list/list_eval_partition.txt', 'r') as f:
    lines = f.readlines()

lines = [line.split() for line in lines]
lines = [(line[0], int(line[1])) for line in lines]

# train_list = []
# valid_list = []
# eval_list = []

# for line in lines:
#     if line[1] == 0:
#         train_list.append(line[0])
#     elif line[1] == 1:
#         valid_list.append(line[0])
#     elif line[1] == 2:
#         eval_list.append(line[0])


with open('datasets/tools/anno_json/celeba_box.json', 'r') as f_box, open('datasets/tools/anno_json/celeba_all.json', 'r') as f_all:
    box_json = json.load(f_box)
    all_json = json.load(f_all)

train_box_json = []
valid_box_json = []
eval_box_json = []

train_all_json = []
valid_all_json = []
eval_all_json = []

for box_data, all_data, line in zip(box_json, all_json, lines):
    assert box_data['file_name'] == all_data['file_name']
    file_name = box_data['file_name']

    assert file_name == line[0]

    # if file_name in train_list:
    if line[1] == 0:
        train_box_json.append(box_data)
        train_all_json.append(all_data)
    # elif file_name in valid_list:
    elif line[1] == 1:
        valid_box_json.append(box_data)
        valid_all_json.append(all_data)
    # elif file_name in eval_list:
    elif line[1] == 2:
        eval_box_json.append(box_data)
        eval_all_json.append(all_data)
    else:
        raise ValueError('Unknown file name: {}'.format(file_name))

assert len(train_box_json) + len(valid_box_json) + len(eval_box_json) == len(box_json)
assert len(train_all_json) + len(valid_all_json) + len(eval_all_json) == len(all_json)

with open('datasets/tools/anno_json/celeba_box_train.json', 'w') as f_box, open('datasets/tools/anno_json/celeba_all_train.json', 'w') as f_all:
    json.dump(train_box_json, f_box)
    json.dump(train_all_json, f_all)

with open('datasets/tools/anno_json/celeba_box_valid.json', 'w') as f_box, open('datasets/tools/anno_json/celeba_all_valid.json', 'w') as f_all:
    json.dump(valid_box_json, f_box)
    json.dump(valid_all_json, f_all)

with open('datasets/tools/anno_json/celeba_box_eval.json', 'w') as f_box, open('datasets/tools/anno_json/celeba_all_eval.json', 'w') as f_all:
    json.dump(eval_box_json, f_box)
    json.dump(eval_all_json, f_all)

