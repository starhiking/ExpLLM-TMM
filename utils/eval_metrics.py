import os
import json
import argparse
import sys
sys.path.append('.')
from datasets.constants import TASK_NAME
# from utils.info_wechat import sc_send

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="checkpoints/ckpts/RAF_DB/eval")
    return parser.parse_args()

def main():
    args = get_args()
    json_path = os.path.join(args.eval_dir, "result_all.json")
    with open(json_path, "r") as f:
        results = json.load(f)
    
    # reset
    tasks = dict()
    for k,v in TASK_NAME.items():
        tasks.update({v:[]})

    print(tasks)

    for res in results:
        task_type = res["task"]
        tasks[task_type].append(res)

    # calculate each metrics
    for k,v in tasks.items():
        print(f"Task: {k}, Total: {len(v)}")

        if k == "emo_estim":
            acc = 0
            dismatch = 0
            emo_id_list = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
            emo_id_dict = {emo_id_list[i]:i for i in range(len(emo_id_list))}
            for res in v:
                # res["pred"] = res["pred"][1:-1] # remove the [], first and last character
                if res["pred"] not in emo_id_list:
                    dismatch += 1 
                    continue
                # assert res["pred"] in emo_id_list, f"pred: {res['pred']}"

                gt_number = emo_id_dict[res["gt"]]
                pred_number = emo_id_dict[res["pred"]]
                if gt_number == pred_number:
                    acc += 1

            acc = acc / len(v)
            print(f"Exp dismatch number: {dismatch} \nExp Accuracy: {acc}")        
            # sc_send('主人服务器训练测试完了', f"Exp dismatch number: {dismatch} \nExp Accuracy: {acc}")

if __name__ == "__main__":
    main()