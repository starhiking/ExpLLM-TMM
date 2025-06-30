import argparse
from transformers import AutoTokenizer, AutoConfig
import torch
import os
import json
from tqdm import tqdm
from transformers import StoppingCriteria
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import pickle as pk
import time

import math
from models import LocLLMModel
from datasets import FACETASKDataset
# from datasets.coco import COCODataset, COCO_KEYPOINT_NAME, KeypointLocationDescription, KeypointLocationQuestion, transform_preds
from datasets.convsersation import conv_face_task #conv_keypoint, conv_llama2, conv_simple
from dataclasses import dataclass
import re
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from datasets.constants import FaceTaskDescription, FaceTaskQuestion, TASK_NAME

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
PREFIX_IMAGE = "Image: "

@dataclass
class DataCollatorForSupervisedDataset(object):
    def __init__(self, image_token_len, conv_format):
        self.image_token_len = image_token_len
        self.conv_format = conv_format


    def __call__(self, instances):
        """Collate examples for supervised fine-tuning."""
        batch_prompts = []
        batch_images = []
        batch_has_images = []
        result_dicts = []

        for i, line in enumerate(instances):
            images = line['image'].unsqueeze(0)
            label = line['label']
            offset = line['offset']

            qtypes = ['bbox'] #['bbox','landmark','attr','headpose','age_gender_race']
            for qtype in qtypes:
                # 单独一轮的 conversation
                if label[qtype] is None:
                    continue

                result_dict = {}
                conv = conv_face_task.copy()
                task_name = TASK_NAME[qtype]
                conv.append_message(conv.roles[0], FaceTaskDescription[task_name])
                conv.append_message(conv.roles[1], FaceTaskQuestion[task_name])
                conv.append_message(conv.roles[2], None)

                text_inputs = conv.get_prompt()
                text_parts = text_inputs.split("\nTASK: ") # 插在 system 内容后面
                assert len(text_parts) == 2, "text_parts should be 2"
                text_inputs = text_parts[0] + "\n" + PREFIX_IMAGE + self.image_token_len * DEFAULT_IMAGE_PATCH_TOKEN + "\nTASK: " + text_parts[1]
                # text_inputs = PREFIX_IMAGE + cur_token_len * DEFAULT_IMAGE_PATCH_TOKEN + "\n" + text_inputs # 调整插入位置
                
                has_images = True

                result_dict['task'] = task_name # task name
                result_dict['gt'] = label[qtype] # result
                result_dict['offset'] = offset # offset for adjust img to center, which is useful for original image
                result_dict['img_path'] = line['img_path'] # img_path
                # result_dict['initial_prompt'] = text_inputs
                batch_prompts.append(text_inputs)
                batch_images.append(images)
                batch_has_images.append(has_images)
                result_dicts.append(result_dict)

        return result_dicts, batch_prompts, batch_images, batch_has_images


@torch.no_grad()
def worker(model, tokenizer, dataset, args, output_dir):
    crop_size = model.config.crop_size
    image_token_len = model.config.num_patches

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    indices = list(range(rank, len(dataset), world_size))
    print("==>" + " Worker {} Started, responsible for {} images".format(rank, len(indices)))

    sub_dataset = torch.utils.data.Subset(dataset, indices)
    batch_size = 8 # 8 的 功率已经340W 了，加到 12 耗时也一样
    data_loader = DataLoader(sub_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=DataCollatorForSupervisedDataset(image_token_len, args.conv_format))

    all_preds = []
    for result_dicts, batch_prompts, batch_images, batch_has_images in tqdm(data_loader):

        tokenized_output = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        batch_images = torch.cat(batch_images, dim=0).cuda() # batch,3,224,224

        input_ids = torch.as_tensor(tokenized_output.input_ids).cuda()
        attention_mask = torch.as_tensor(tokenized_output.attention_mask).cuda()

        with torch.inference_mode():
            output_dict = model.generate(
                input_ids,
                images=batch_images,
                has_images=batch_has_images,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=100, # why setting 13
                output_scores=True,
                return_dict_in_generate=True
            )
            output_ids = output_dict['sequences']
            output_scores = output_dict['scores']

        # 单个解码，改成一批次解码
        # outputs = []
        # for input_id, output_id in zip(input_ids, output_ids):
        #     input_token_len = input_id.shape[0]
        #     n_diff_input_output = (input_id != output_id[:input_token_len]).sum().item()
        #     if n_diff_input_output > 0:
        #         print(f'[Warning] Sample: {n_diff_input_output} output_ids are not the same as the input_ids')
        #     output = tokenizer.batch_decode(output_id[input_token_len:].unsqueeze(0), skip_special_tokens=True)[0]
        #     output = output.strip()
        #     print(output, result_dicts[0]['result'])
        #     outputs.append(output)
        
        ## 可以改，因为任务一样，input_ids 也一样，所以只需要一个就行
        assert input_ids[0].equal(input_ids[-1]), "input_ids should be the same"
        input_id = input_ids[0]
        input_token_len = input_id.shape[0]
        n_diff_input_output = (input_id != output_ids[0][:input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample: {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:,input_token_len:], skip_special_tokens=True)
        outputs = [output.strip() for output in outputs]

        # print(outputs)
        for pred, gt in zip(outputs, result_dicts):
            pred_gt = gt.copy()
            pred_gt.update({'pred': pred})
            all_preds.append(pred_gt)
    
    # save results
    sub_result_dir = os.path.join(output_dir,"sub_results")
    if not os.path.exists(sub_result_dir):
        os.makedirs(sub_result_dir, exist_ok=True)
    with open(os.path.join(sub_result_dir, f'result_{rank}.json'), 'w') as f:
        json.dump(all_preds, f)

    print("==>" + " Worker {} Finished".format(rank))
    torch.distributed.barrier()

    if rank == 0:
        # manually sleep to wait all file are saved
        while True:
            ready = True
            for r in range(world_size):
                if not os.path.exists(os.path.join(sub_result_dir, f'result_{r}.json')):
                    ready = False
            if ready: 
                break
            else:
                time.sleep(20)
        time.sleep(20)
        all_preds = []
        for r in range(world_size):
            with open(os.path.join(sub_result_dir, f'result_{r}.json'), 'r') as fr:
                all_preds.extend(json.load(fr))
        with open(os.path.join(output_dir, 'result_all.json'), 'w') as f:
            json.dump(all_preds, f, indent=4)

        # evaluate
        # TODO: add evaluation code

def eval_model(args):
    torch.distributed.init_process_group(backend='nccl')
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    print('Init process group: world_size: {}, rank: {}'.format(world_size, rank))
    torch.cuda.set_device(rank)

    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left')

    model = LocLLMModel.from_pretrained(model_name, use_cache=True)
    for name, param in model.model.named_parameters():
        if "lora_" not in name:
            param.data = param.data.bfloat16()
    model.lm_head.to(torch.bfloat16)
    model = model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)

    dataset = FACETASKDataset(tokenizer=None,
                        data_path=os.path.join(args.question_file),
                        multimodal_cfg=dict(
                            image_size=224,
                            crop_size=224,
                            conv_format=args.conv_format),
                            is_train=False)

    worker(model, tokenizer, dataset, args, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="checkpoints/ckpts/facetask_pretrain_on_ALL_data")
    parser.add_argument("--question-file", type=str, default="datasets/tools/training_list/test_facedetect.txt")
    # parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-format", type=str, default="facetask")
    parser.add_argument("--output-dir", type=str, default="checkpoints/ckpts/facetask_pretrain_on_ALL_data/eval")
    args = parser.parse_args()

    eval_model(args)
