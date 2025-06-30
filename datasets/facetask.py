import transformers
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import logging
import random
from typing import Dict
import os
import numpy as np
import cv2
import json

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
PREFIX_IMAGE = "Image: "
PREFIX_NO_IMAGE = "Image: N/A"
BEGIN_DESCRIPTION = "<des>"
END_DESCRIPTION = "</des>"
IGNORE_INDEX = -100
DEFAULT_EOS_TOKEN = "</s>"

from datasets.convsersation import conv_face_task
from datasets.constants import TASK_NAME, FaceTaskDescription, FaceTaskQuestion

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
PREFIX_IMAGE = "Image: "
PREFIX_NO_IMAGE = "Image: N/A"
BEGIN_DESCRIPTION = "<des>"
END_DESCRIPTION = "</des>"
IGNORE_INDEX = -100
DEFAULT_EOS_TOKEN = "</s>"
BEGIN_OPTIONS = "<opt>"
END_OPTIONS = "</opt>"
BEGIN_LOC = "<loc>"
END_LOC = "</loc>"
BEGIN_QUESTION = "<qes>"
END_QUESTION = "</qes>"

def face_task_anno_read(data_path:str):
    assert os.path.exists(data_path), "data path not exists"
    with open(data_path, "r") as f:
        file_list = f.readlines()
        file_list = [x.strip() for x in file_list]
    f.close()

    json_list = []
    for files in file_list:
        json_file = files.split(',')[0]
        image_folder = files.split(',')[1]
        repeat_time = float(files.split(',')[2]) if len(files.split(',')) > 2 else 1
        assert os.path.exists(json_file), f"{json_file} json file not exists"
        assert os.path.exists(image_folder), f"{image_folder} image folder not exists"
        json_data = json.load(open(json_file, "r"))
        for data in json_data:
            data.update({'file_name': os.path.join(image_folder, data['file_name'])})
        if repeat_time < 1:
            num = int(len(json_data) * repeat_time)
            json_data = random.sample(json_data, num)
        else:
            repeat_time = int(repeat_time)
            json_data = json_data * repeat_time
        json_list += json_data
    return json_list

class FACETASKDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict,
                 is_train=True
                 ):
        super(FACETASKDataset, self).__init__()
        logging.warning("Loading data...")
        self.size = 224
        list_data_dict = face_task_anno_read(data_path)

        logging.warning("The number of training samples is {}".format(len(list_data_dict)))
        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg
        self.conv_format = self.multimodal_cfg.get("conv_format", "face_task")
        self.question_index = self.multimodal_cfg.get("question_index", 1)

        print('Use Conv Format ', self.conv_format)
        if 'data_augmentation' in self.multimodal_cfg.keys():
            self.data_aug = self.multimodal_cfg['data_augmentation']
        else:
            self.data_aug = False
        if self.multimodal_cfg.get('dino_norm', False):
            norm_mean = (0.485, 0.456, 0.406)
            norm_std = (0.229, 0.224, 0.225)
        else:
            norm_mean = (0.48145466, 0.4578275, 0.40821073)
            norm_std = (0.26862954, 0.26130258, 0.27577711)
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )
        self.emo_train_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
                    transforms.RandomGrayscale(p=0.2),
                ]),
                transforms.RandomApply([
                    transforms.RandomRotation(5),
                    transforms.RandomChoice([
                        transforms.RandomResizedCrop(224, scale=(0.8, 1), ratio=(0.75, 1.3333)),
                        transforms.RandomCrop(224, padding=12),
                    ]),
                ], p=0.5),
                transforms.RandomApply([
                    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), 
                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
                ], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
                transforms.RandomErasing(scale=(0.05,0.12)),
            ]
        )

        self.is_train = is_train

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        if self.is_train:
            return self._parse_data_item(i)
        else:
            return self._parse_data_item_val(i)

    def _parse_data_item_val(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = {}
        
        image, label, offset = self._get_image_target(sources)
        return dict(image=image, label=label, offset=offset, img_path=sources['file_name'])

    def _parse_data_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = {}
        
        image, label, _ = self._get_image_target(sources)
        data_dict['image'] = image
        data_dict['has_image'] = True
        data_dict['img_path'] = sources['file_name']
        cur_token_len = 256

        # design conversation
        # question_index: only detect bbox,  first box + random, random

        task_des = []
        task_qs = []
        task_ans = []

        qtypes = random_question_index(self.question_index)
        for q_type in qtypes:
            if label[q_type] is None:
                continue
            task_name = TASK_NAME[q_type]
            task_des.append(FaceTaskDescription[task_name])
            task_qs.append(FaceTaskQuestion[task_name])
            # # landmark 和 bbox 和 headpose 需要处理成固定长度, 注意 bbox 和 landmark 有可能存在负数，符号位会占用一个字符 TODO
            if q_type == 'landmark':
                # landmark 5 points [[x,y],[],[],[],[]]
                landmark = label[q_type]
                formatted_list = [[f"{landmark_x[0]:.4f}",f"{landmark_x[1]:.4f}"] for landmark_x in landmark]
                formatted_str = f"{formatted_list}".replace("'",'')
                task_ans.append(formatted_str)
            elif q_type == 'bbox':
                bbox = label[q_type]
                formatted_list = [f"{num:.4f}" for num in bbox]
                formatted_str = f"{formatted_list}".replace("'",'')
                task_ans.append(formatted_str)
            elif q_type == 'headpose':
                # 3 degrees [,,]
                headpose = label[q_type]
                formatted_list = [f"{num:.4f}" for num in headpose]
                formatted_str = f"{formatted_list}".replace("'",'')
                task_ans.append(formatted_str)
            # elif q_type == 'emo':
            #     task_ans.append('[' + f"{label[q_type]}" + ']') # add [] for emo
            else:
                task_ans.append(f"{label[q_type]}")

            # break # 只有一个问题

        conv = conv_face_task.copy()       
        assert len(conv.messages) == 0, "conversation should be empty"
        for task_des_i, task_q_i, task_ans_i in zip(task_des, task_qs, task_ans):
            if self.conv_format == 'face_task':
                conv.append_message(conv.roles[0], task_des_i)
                conv.append_message(conv.roles[1], task_q_i)
                conv.append_message(conv.roles[2], task_ans_i)

        text_inputs = conv.get_prompt()
        text_parts = text_inputs.split("\nTASK: ") # 插在 system 内容后面
        assert len(text_parts) == 2, "text_parts should be 2"
        text_inputs = text_parts[0] + "\n" + PREFIX_IMAGE + cur_token_len * DEFAULT_IMAGE_PATCH_TOKEN + "\nTASK: " + text_parts[1]
        # text_inputs = PREFIX_IMAGE + cur_token_len * DEFAULT_IMAGE_PATCH_TOKEN + "\n" + text_inputs # TODO 调整插入位置

        inputs = self.tokenizer(text_inputs,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True).input_ids[0]
        
        target = inputs.clone()
        sep = conv.sep1 + conv.roles[2] + ": "
        rounds = text_inputs.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(self.tokenizer(rou).input_ids)
            instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2   # <s> <space>
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        data_dict.update(
            dict(input_ids=inputs,
                labels=target)
        )

        return data_dict
    
    def _get_image_target(self, sources):
        file_name = sources['file_name']
        image_file = file_name # os.path.join(image_folder, file_name)
        assert os.path.exists(image_file), f"Image file {image_file} not exists"
        
        image = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        bbox = sources.get('bbox', None)
        landmark = sources.get('landmark', None)
        attr = sources.get('attr', None)
        headpose = sources.get('headpose', None)
        age_gender_race = sources.get('age_gender_race', None)
        emo = sources.get('emo', None)
        description = sources.get('description', None)

        # adjust the image size, and corresponding bbox, landmark
        landmark, bbox, offset = adjust_bbox_landmark(image, landmark, bbox) # TODO 可能存在负数的情况，这会导致 token 长度不一致
        image = expand2square(image)

        # resize and normalize the bbox and landmark
        if landmark is not None:
            landmark = [[float(f"{landmark_x / image.shape[1] :.4f}"), float(f"{landmark_y / image.shape[0] :.4f}") ] for landmark_x, landmark_y in landmark]
        if bbox is not None:
            bbox = [float(f"{x / image.shape[0] :.4f}") for x in bbox]
        
        # # # TODO visualize bbox and landmarks
        # image_copy = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        # if bbox is not None:
        #     bbox = [int(x*image.shape[0]) for x in bbox]
        #     x, y, w, h = bbox
        #     cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # if landmark is not None:
        #     landmark = [[int(x*image.shape[0]) for x in y] for y in landmark]
        #     for x, y in landmark:
        #         cv2.circle(image_copy, (x, y), 2, (0, 0, 255), -1)
        # cv2.imwrite(f'test_samples/{os.path.basename(file_name)}', image_copy)

        image = cv2.resize(image, (self.size, self.size))
        # image = self.transforms(image) # grounding data may not need augmentation
        img_PIL = transforms.ToPILImage()(image)
        image = self.emo_train_transforms(img_PIL) if self.data_aug else self.transforms(img_PIL)
        
        ## visualize the augmented image
        # rgb_image = ((image.permute(1, 2, 0).numpy()  * np.array([0.26862954, 0.26130258, 0.27577711]) + np.array([0.48145466, 0.4578275, 0.40821073])) * 255).astype(np.uint8)
        # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f'test_samples/{os.path.basename(file_name)}', bgr_image)

        label = dict({
            'bbox': bbox,
            'landmark': landmark,
            'attr': attr,
            'headpose': headpose,
            'age_gender_race': age_gender_race,
            'emo': emo,
            'description': description
        })

        return image, label, offset


def expand2square(cv_img):
    background_color = [123, 117, 104]  # Convert [0.48145466, 0.4578275, 0.40821073] to [123, 117, 104]
    height, width, channels = cv_img.shape
    
    if width == height:
        return cv_img
    elif width > height:
        result = np.full((width, width, channels), background_color, dtype=np.uint8)
        result[(width - height) // 2:(width + height) // 2, :] = cv_img
        return result
    else:
        result = np.full((height, height, channels), background_color, dtype=np.uint8)
        result[:, (height - width) // 2:(height + width) // 2] = cv_img
        return result

def adjust_bbox_landmark(cv_img, landmark, bbox):
    height, width, _ = cv_img.shape
    if height == width:
        return landmark, bbox, dict(x=0, y=0)
    
    x_offset = 0
    y_offset = 0
    if height > width:
        x_offset = (height - width) // 2
    else:
        y_offset = (width - height) // 2
    
    new_bbox = None
    if bbox is not None:
        new_bbox = [bbox[0] + x_offset, bbox[1] + y_offset, bbox[2], bbox[3]]
    
    offset = dict(x=x_offset, y=y_offset)

    new_landmark = None
    if landmark is not None:
        new_landmark = []
        for landmark_i in landmark:
            new_landmark.append([landmark_i[0] + x_offset, landmark_i[1] + y_offset])
    
    return new_landmark, new_bbox, offset

def random_question_index(question_index:int):
    q_index = list(TASK_NAME.keys()) #['bbox','landmark','attr','headpose','age_gender_race', 'emo']
    select_index = []
    if question_index == 0:
        # only detect bbox
        select_index.append(q_index[0])
    elif question_index == 1:
        # first box + random other
        select_index.append(q_index[0])
        select_index += random.sample(q_index[1:], len(q_index[1:]))
    elif question_index == 2:
        # random all
        select_index = random.sample(q_index, len(q_index))
    elif question_index == -1:
        select_index = q_index

    return select_index




############# debug #############
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
# FIXME: seems wrong?
# DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    llama_path: Optional[str] = field(default="")
    dino_path: Optional[str] = field(default=None)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=True)
    freeze_vit: bool = field(default=True)
    freeze_llm: bool = field(default=True)
    save_mm_projector: bool = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    image_token_len: int = 0
    image_size: int = field(default=224)
    crop_size: int = field(default=224)
    data_augmentation: bool = field(default=False)
    conv_format: str = field(default="keypoint")
    question_index: int = field(default=1) # 1 for random one, 0 for only box, 2 for box + random

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

@dataclass
class LoRAArguments:
    lora_vision_r: int = field(default=8)
    lora_vision_alpha: float = field(default=16)
    lora_vision_dropout: float = field(default=0.05)
    lora_vision_enable: bool = field(default=False)
    lora_llm_r: int = field(default=8)
    lora_llm_alpha: float = field(default=16)
    lora_llm_dropout: float = field(default=0.05)
    lora_llm_enable: bool = field(default=False)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            assert all(x is not None and x.shape == images[0].shape for x in images)
            batch['images'] = torch.stack(images)

        assert 'has_image' in instances[0].keys()
        has_images = [instance['has_image'] for instance in instances]
        batch['has_images'] = has_images

        return batch
    
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = FACETASKDataset 
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                multimodal_cfg=dict(
                                    data_augmentation=data_args.data_augmentation,
                                    image_size=data_args.image_size,
                                    crop_size=data_args.crop_size,
                                    conv_format=data_args.conv_format,
                                    question_index=data_args.question_index,))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


if __name__ == '__main__':
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoRAArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    train_dataset = data_module['train_dataset']
    data_collator = data_module['data_collator']
    for i in range(len(train_dataset)):
        data = train_dataset[i]
        batch = data_collator([data])
        print(batch)
        