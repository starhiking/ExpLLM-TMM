import os
import json

# 读取 pred_emo 文件
with open(os.path.join("checkpoints/ckpts/RAFDB_91.03/eval/result_all.json"), 'r') as f:
    pred_emo = json.load(f)

# 读取 standard_des 文件
with open(os.path.join("data_list/anno_json/rafdb_emo_au_val_mini_description.json"), 'r') as f:
    standard_des = json.load(f)

# 排序
pred_emo_sort = sorted(pred_emo, key=lambda x: x['img_path'])
standard_des_sort = sorted(standard_des, key=lambda x: x['file_name'])

for pred_emo_i, standard_des_i in zip(pred_emo_sort, standard_des_sort):
    len_str = len(standard_des_i['file_name'])
    assert(pred_emo_i['img_path'][-len_str:] == standard_des_i['file_name'])
    standard_des_i["gt_emo"] = standard_des_i["emo"]
    if standard_des_i["emo"] != pred_emo_i["pred"]:
        print(f"change {standard_des_i['file_name']} from {standard_des_i['emo']} to {pred_emo_i['pred']}")
        standard_des_i["emo"] = pred_emo_i["pred"]

with open("need_test_raf-db_91.03_des.json", 'w') as f:
    json.dump(standard_des_sort,f,indent=4)
