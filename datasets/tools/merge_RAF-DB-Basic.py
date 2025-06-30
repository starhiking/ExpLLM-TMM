import os
import json
import numpy as np
import cv2

scrfd_json = "datasets/tools/scrfd_results/RAF-DB-Basic_results.json"

raf_db_path = "/mnt/data/lanxing/RAF-DB/basic/Annotation"
img_folder_path = "/mnt/data/lanxing/RAF-DB/basic/Image/"

box_folder_path = os.path.join(raf_db_path, "boundingbox")
landmark_folder_path = os.path.join(raf_db_path, "manual")

merge_bbox_file_path = "datasets/tools/scrfd_results/list_bbox_raf-db-basic_new.txt"

scrfd_json_data = json.load(open(scrfd_json, "r"))


with open(merge_bbox_file_path, "w") as f:
    f.write(f"{len(scrfd_json_data)}\n")
    f.write("image_id x_1 y_1 width height\n")

    for scrfd_data in scrfd_json_data:
        file_name = scrfd_data["path"]
        box_file_name = os.path.join(box_folder_path, file_name.replace(".jpg", "_boundingbox.txt"))
        landmark_file_name = os.path.join(landmark_folder_path, file_name.replace(".jpg", "_manu_attri.txt"))
        
        assert os.path.exists(box_file_name), f"{box_file_name} not exists"
        assert os.path.exists(landmark_file_name), f"{landmark_file_name} not exists"

        with open(box_file_name, "r") as f_box:
            # x1, y1, x2, y2
            naive_box = f_box.readlines()[0].strip().split()
            naive_box = [int(float(x)) for x in naive_box]
            
            # convert to x1, y1, w, h
            w = naive_box[2] - naive_box[0]
            h = naive_box[3] - naive_box[1]
            naive_box = [naive_box[0], naive_box[1], w, h]

        with open(landmark_file_name, "r") as f_landmark:
            naive_landmarks = f_landmark.readlines()[:5]
            naive_landmarks = [landmark.strip().split() for landmark in naive_landmarks]
            naive_landmarks = [[int(float(x)), int(float(y))] for x,y in naive_landmarks]
        
        if len(scrfd_data['result']) == 0:
            img_path = os.path.join(img_folder_path, "original", file_name)
            assert os.path.exists(img_path), f"{img_path} not exists"
            img = cv2.imread(img_path)
            for (x,y) in naive_landmarks:
                cv2.circle(img, (x,y), 2, (0,255,0), 2) # naive label, green
            cv2.rectangle(img, (naive_box[0], naive_box[1]), (naive_box[0]+naive_box[2], naive_box[1]+naive_box[3]), (0,255,0), 2) # naive label
            cv2.imwrite(f"vis_samples/{file_name}", img)

            print(f"{file_name} no face detected, use the original naive_box")
            f.write(f"{file_name} {naive_box[0]} {naive_box[1]} {naive_box[2]} {naive_box[3]}\n")
            continue

        # calculate the distance between the five landmark pairs
        min_distance = 3000 # initialize the distance to a large number
        min_face_id = 'face_1'
        for faces in scrfd_data['result']:
            for face_id, face_value in faces.items():
                face_landmark = face_value['landmarks']
                face_landmark_list = [face_landmark['left_eye'][0],face_landmark['left_eye'][1],face_landmark['right_eye'][0],face_landmark['right_eye'][1],face_landmark['nose'][0],face_landmark['nose'][1], \
                                    face_landmark['mouth_left'][0],face_landmark['mouth_left'][1],face_landmark['mouth_right'][0],face_landmark['mouth_right'][1]]

                face_landmark_list = [[int(float(x)), int(float(y))] for x,y in zip(face_landmark_list[0::2], face_landmark_list[1::2])]

                distance = np.linalg.norm(np.array(naive_landmarks) - np.array(face_landmark_list))
                if distance < min_distance:
                    min_distance = distance
                    min_face_id = face_id
                    accurate_landmark = face_landmark_list
        
        assert min_face_id in scrfd_data['result'][int(min_face_id.split('_')[1])-1], f"{min_face_id} not in {file_name}"
        scrfd_box = scrfd_data['result'][int(min_face_id.split('_')[1])-1][min_face_id]['facial_area']
        scrfd_box = [int(float(x)) for x in scrfd_box]
        face_landmark_list = accurate_landmark

        ## check naive landmarks in scrfd box, at least four points in the box
        point_flag = 0
        for x,y in naive_landmarks:
            if scrfd_box[0] < x < scrfd_box[0]+scrfd_box[2] and scrfd_box[1] < y < scrfd_box[1]+scrfd_box[3]:
                point_flag += 1    

        if point_flag < 4:
            img_path = os.path.join(img_folder_path, "original", file_name)
            assert os.path.exists(img_path), f"{img_path} not exists"
            img = cv2.imread(img_path)
            for (x,y), (x2,y2) in zip(naive_landmarks, face_landmark_list):
                cv2.circle(img, (x,y), 2, (0,255,0), 2) # naive label, green
                cv2.circle(img, (x2,y2), 2, (255,0,0), 2) # scrfd result, blue
            cv2.rectangle(img, (naive_box[0], naive_box[1]), (naive_box[0]+naive_box[2], naive_box[1]+naive_box[3]), (0,255,0), 2) # naive label
            cv2.rectangle(img, (scrfd_box[0], scrfd_box[1]), (scrfd_box[0]+scrfd_box[2], scrfd_box[1]+scrfd_box[3]), (255,0,0), 2) # scrfd result
            cv2.imwrite(f"vis_samples/{file_name}", img)
            
            # 如果关键点溢出了scrfd的box，使用naive的box
            print(f"{file_name} landmarks out of scrfd box, use the original bbox")
            f.write(f"{file_name} {naive_box[0]} {naive_box[1]} {naive_box[2]} {naive_box[3]}\n")

        else:
            # 匹配成功，使用scrfd的结果
            f.write(f"{file_name} {scrfd_box[0]} {scrfd_box[1]} {scrfd_box[2]} {scrfd_box[3]}\n")

