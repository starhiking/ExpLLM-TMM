import os
import json
import numpy as np
import cv2

scrfd_json = "datasets/tools/scrfd_results/CelebA_results.json"

celeba_path = "/mnt/data/lanxing/celeba"

box_file_path = os.path.join(celeba_path, "Anno/list_bbox_celeba.txt")
landmark_file_path = os.path.join(celeba_path, "Anno/list_landmarks_celeba.txt")

merge_bbox_file_path = "datasets/tools/scrfd_results/list_bbox_celeba_new.txt"

scrfd_json_data = json.load(open(scrfd_json, "r"))

with open(box_file_path, "r") as f:
    lines = f.readlines()
    head_lines = lines[:2]
    box_lines = lines[2:]
f.close()

with open(landmark_file_path, "r") as f:
    lines = f.readlines()
    landmark_lines = lines[2:]
f.close()

with open(merge_bbox_file_path, "w") as f:
    f.write(head_lines[0])
    f.write(head_lines[1])

    for scrfd_data, celeb_box, landmark in zip(scrfd_json_data, box_lines, landmark_lines):
        box = celeb_box.strip().split()
        landmark = landmark.strip().split()

        assert box[0] == landmark[0] == scrfd_data["path"], f"{box[0]} file name not match"

        if len(scrfd_data['result']) == 0:
            print(f"{box[0]} no face detected, use the original bbox")
            f.write(celeb_box)
            continue

        # calculate the distance between the five landmark pairs
        celeba_landmark = [[int(x), int(y)] for x,y in zip(landmark[1::2], landmark[2::2])]
        min_distance = 3000 # initialize the distance to a large number
        min_face_id = 'face_1'
        for faces in scrfd_data['result']:
            for face_id, face_value in faces.items():
                face_landmark = face_value['landmarks']
                face_landmark_list = [face_landmark['left_eye'][0],face_landmark['left_eye'][1],face_landmark['right_eye'][0],face_landmark['right_eye'][1],face_landmark['nose'][0],face_landmark['nose'][1], \
                                    face_landmark['mouth_left'][0],face_landmark['mouth_left'][1],face_landmark['mouth_right'][0],face_landmark['mouth_right'][1]]

                face_landmark_list = [[int(float(x)), int(float(y))] for x,y in zip(face_landmark_list[0::2], face_landmark_list[1::2])]

                distance = np.linalg.norm(np.array(celeba_landmark) - np.array(face_landmark_list))
                if distance < min_distance:
                    min_distance = distance
                    min_face_id = face_id
        
        assert min_face_id in scrfd_data['result'][int(min_face_id.split('_')[1])-1], f"{min_face_id} not in {box[0]}"
        scrfd_box = scrfd_data['result'][int(min_face_id.split('_')[1])-1][min_face_id]['facial_area']
        scrfd_box = [int(float(x)) for x in scrfd_box]
        celeba_box = [int(x) for x in box[1:]]

        ## check celebA landmarks in scrfd box, at least four points in the box
        point_flag = 0
        for x,y in celeba_landmark:
            if scrfd_box[0] < x < scrfd_box[0]+scrfd_box[2] and scrfd_box[1] < y < scrfd_box[1]+scrfd_box[3]:
                point_flag += 1    

        if point_flag < 4:
            img = cv2.imread(os.path.join(f"{celeba_path}/img_celeba", box[0]))
            for (x,y), (x2,y2) in zip(celeba_landmark, face_landmark_list):
                cv2.circle(img, (x,y), 2, (0,255,0), 2) # celeba label, green
                cv2.circle(img, (x2,y2), 2, (255,0,0), 2) # scrfd result, blue
            cv2.rectangle(img, (celeba_box[0], celeba_box[1]), (celeba_box[0]+celeba_box[2], celeba_box[1]+celeba_box[3]), (0,255,0), 2) # celeba label
            cv2.rectangle(img, (scrfd_box[0], scrfd_box[1]), (scrfd_box[0]+scrfd_box[2], scrfd_box[1]+scrfd_box[3]), (255,0,0), 2) # scrfd result
            cv2.imwrite(f"{celeba_path}/vis_samples/{box[0]}", img)
            
            # 如果关键点溢出了scrfd的box，使用celeba的box
            print(f"{box[0]} landmarks out of scrfd box, use the original bbox")
            f.write(celeb_box)

        else:
            # 默认使用scrfd的结果
            f.write(f"{box[0]} {scrfd_box[0]} {scrfd_box[1]} {scrfd_box[2]} {scrfd_box[3]}\n")

