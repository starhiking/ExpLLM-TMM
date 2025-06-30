# 转化出2种形式
# 1. 只有 bounding box 的形式
# 2. 有 bounding box 和 其他属性的形式

# bbox: x1, y1, w, h
# landmark: left_eye[x1, y1], right_eye[x2, y2], nose[x3, y3], left_mouth[x4, y4], right_mouth[x5, y5]

import os
import json

celeba_path = "/mnt/data/lanxing/celeba"

# box_file_path = os.path.join(celeba_path, "Anno/list_bbox_celeba.txt")
box_file_path = "datasets/tools/scrfd_results/list_bbox_celeba_new.txt"
landmark_file_path = os.path.join(celeba_path, "Anno/list_landmarks_celeba.txt")
attr_file_path = os.path.join(celeba_path, "Anno/list_attr_celeba.txt")

assert os.path.exists(box_file_path), "box file not exists"
assert os.path.exists(landmark_file_path), "landmark file not exists"
assert os.path.exists(attr_file_path), "attr file not exists"

with open(box_file_path, "r") as f:
    lines = f.readlines()
    box_lines = lines[2:]
f.close()

with open(landmark_file_path, "r") as f:
    lines = f.readlines()
    landmark_lines = lines[2:]
f.close()

with open(attr_file_path, "r") as f:
    lines = f.readlines()
    attr_lines = lines[2:]
f.close()

flag_head = False

with open(os.path.join('datasets/tools/anno_json', "celeba_box.json"), "w") as f_box, open(os.path.join('datasets/tools/anno_json', "celeba_all.json"), "w") as f_all:
    for box, landmark, attr in zip(box_lines, landmark_lines, attr_lines):

        if not flag_head:
            f_box.write("[")
            f_all.write("[")
            flag_head = True
        else:
            f_box.write(",\n")
            f_all.write(",\n")

        box = box.strip().split()
        landmark = landmark.strip().split()
        attr = attr.strip().split()

        assert box[0] == landmark[0] == attr[0], f"{box[0]} file name not match"

        file_name = box[0]
        box = [int(x) for x in box[1:]]
        landmark = [[int(x), int(y)] for x,y in zip(landmark[1::2], landmark[2::2])]
        attr = [int(int(x) == 1) for x in attr[1:]] # convert -1 to 0

        box_dict = {
            "file_name": file_name,
            "bbox": box
        }

        all_dict = box_dict.copy()
        all_dict.update({
            "landmark": landmark,
            "attr": attr            
            })

        f_box.write(json.dumps(box_dict))
        f_all.write(json.dumps(all_dict))
    
    f_box.write("]")
    f_all.write("]")
