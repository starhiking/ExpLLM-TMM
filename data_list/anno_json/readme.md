# json 文件介绍

box 是用 scrfd 跑出来的， 如果原来标签中就存在 box， 则用原来的 box。

如果原来的图像中存在多个人脸，且自带 box，则直接用原来的 box。

其他情况进行匹配，匹配不成功会被过滤掉，因为会存在语义模糊。

数据集来源尽可能是 wild 数据集

## 300wlp & aflw

- 300wlp_all.json 是 300wlp 和 AFLW 数据集的标注文件, 122,450, 包含了bbox (x,y,w,h) 和 headpose。

- 300wlp_box.json 是 300wlp 和 AFLW 数据集的标注文件，122,450, 包含了bbox (x,y,w,h)。

- aflw2000_all.json 是 AFLW 测试数据集的标注文件，2,000, 包含了bbox (x,y,w,h) 和 headpose。

- aflw2000_box.json 是 AFLW  测试数据集的标注文件，2,000, 包含了bbox (x,y,w,h)。

其中 headpose 的处理方法是， 参考了[hopenet](https://github.com/natanielruiz/deep-head-pose/blob/master/code/datasets.py#L95)
        
        pose_params = pre_pose_params[:3]
        pitch = round(pose_params[0] * 180 / np.pi, 4)
        yaw = round(pose_params[1] * 180 / np.pi, 4)
        roll = round(pose_params[2] * 180 / np.pi, 4)

## AffectNet

- Affectnet_280k_7cls_train.json 是 AffectNet 训练数据集的标注文件，283,901, 包含了7种expression。

- Affectnet_280k_7cls_val.json 是 AffectNet 测试数据集的标注文件，3,500, 包含了7种expression。

- Affectnet_280k_8cls_train.json 是 AffectNet 训练数据集的标注文件，287,651, 包含了8种expression。

- Affectnet_280k_8cls_val.json 是 AffectNet 测试数据集的标注文件，4,000, 包含了8种expression。

- Affectnet_kaggle_train-sample-affectnet.json 是 AffectNet kaggle提供的训练数据集的标注文件，37,553, 包含了8种expression。

- Affectnet_kaggle_valid-sample-affectnet.json 是 AffectNet kaggle提供的测试数据集的标注文件，4,000, 包含了8种expression。

- Affectnet_kaggle_train-sample-affectnet_mini_description.json 是 AffectNet kaggle提供的训练数据集的标注文件，37,553, 包含了8种expression， au值（预测的）和表情的描述（规范字数 130 words， 推荐使用）。

- Affectnet_kaggle_valid-sample-affectnet_mini_description.json 是 AffectNet kaggle提供的测试数据集的标注文件，4,000, 包含了8种expression， au值（预测的）和表情的描述（规范字数 130 words， 推荐使用）。

## celeba （wild，没裁剪前的）

- celeba_all.json 是 celeba 所有数据集的标注文件，202,599, 包含了bbox (x,y,w,h) 和 5个landmark， 40个属性。

- celeba_all_train.json 是 celeba 训练数据集的标注文件，162,770, 包含了bbox (x,y,w,h) 和 5个landmark， 40个属性。

- celeba_all_valid.json 是 celeba 验证数据集的标注文件，19,867, 包含了bbox (x,y,w,h) 和 5个landmark， 40个属性。

- celeba_all_eval.json 是 celeba 测试数据集的标注文件，19,962, 包含了bbox (x,y,w,h) 和 5个landmark， 40个属性。

- celeba_box.json 是 celeba 所有数据集的标注文件，202,599, 包含了bbox (x,y,w,h)。

- celeba_box_train.json 是 celeba 训练数据集的标注文件，162,770, 包含了bbox (x,y,w,h)。

- celeba_box_valid.json 是 celeba 验证数据集的标注文件，19,867, 包含了bbox (x,y,w,h)。

- celeba_box_eval.json 是 celeba 测试数据集的标注文件，19,962, 包含了bbox (x,y,w,h)。

## raf-db （raf-db 自带的人脸框裁剪结果跟 scrfd 非常接近，直接使用）

- rafdb_train.json 是 raf-db 训练数据集的标注文件，12,271, 包含了7种expression。

- rafdb_valid.json 是 raf-db 验证数据集的标注文件，3,068, 包含了7种expression。

- rafdb_emo_au_train_description.json 是 raf-db 训练数据集的标注文件，12,271, 包含了7种expression，au值和表情的描述 (没有规范字数，有点长)。

- rafdb_emo_au_valid_description.json 是 raf-db 验证数据集的标注文件，3,068, 包含了7种expression，au值和表情的描述 (没有规范字数，有点长)。

- rafdb_emo_au_train_mini_description.json 是 raf-db 训练数据集的标注文件，12,271, 包含了7种expression，au值（预测的）和表情的描述（规范字数 130 words， 推荐使用）。

- rafdb_emo_au_valid_mini_description.json 是 raf-db 验证数据集的标注文件，3,068, 包含了7种expression，au值（预测的）和表情的描述（规范字数 130 words， 推荐使用）。

## fairface （1.25 的数据，算 wild 了）

- fairface_all.json 是 fairface 所有数据集的标注文件，69,639, 包含了bbox (x,y,w,h) 和 7种race， 7种age， 2种gender。

- fairface_all_train.json 是 fairface 训练数据集的标注文件，61,830, 包含了bbox (x,y,w,h) 和 7种race， 7种age， 2种gender。

- fairface_all_valid.json 是 fairface 测试数据集的标注文件，7,809, 包含了bbox (x,y,w,h) 和 7种race， 7种age， 2种gender。

- fairface_box.json 是 fairface 所有数据集的标注文件，69,639, 包含了bbox (x,y,w,h)。

- fairface_box_train.json 是 fairface 训练数据集的标注文件，61,830, 包含了bbox (x,y,w,h)。

- fairface_box_valid.json 是 fairface 测试数据集的标注文件，7,809, 包含了bbox (x,y,w,h)。

其中age， gender， race 做了map处理，具体如下：

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

## utkface （wild，没裁剪前的）

- utkface_all.json 是 utkface 所有数据集的标注文件，22,308, 包含了bbox (x,y,w,h) 和 3种race， 2种gender， 5种age。

- utkface_all_train.json 是 utkface 训练数据集的标注文件，17,846, 包含了bbox (x,y,w,h) 和 3种race， 2种gender， 5种age。

- utkface_all_valid.json 是 utkface 验证数据集的标注文件，4,462, 包含了bbox (x,y,w,h) 和 3种race， 2种gender， 5种age。

- utkface_box.json 是 utkface 所有数据集的标注文件，22,308, 包含了bbox (x,y,w,h)。

- utkface_box_train.json 是 utkface 训练数据集的标注文件，17,846, 包含了bbox (x,y,w,h)。

- utkface_box_valid.json 是 utkface 验证数据集的标注文件，4,462, 包含了bbox (x,y,w,h)。

其中 age 和 gender 做了处理，race没变，具体如下：

        # convert age: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+ to 0,1,2,....,8
        age = 0 + int(age>=3) + int(age>=10) + int(age>=20) + int(age>=30) + int(age>=40) + int(age>=50) + int(age>=60) + int(age>=70)

        # convert gender
        gender = int(gender == 0) # 1 for male, 0 for female
