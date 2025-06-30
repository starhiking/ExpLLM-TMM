import os
import json
import numpy as np
import cv2
from scipy import io

scrfd_json = "datasets/tools/scrfd_results/300WLP_results.json"

folder_path = "/mnt/data/lanxing/300W_LP"

merge_bbox_file_path = "datasets/tools/scrfd_results/list_bbox_300wlp_new.txt"

scrfd_json_data = json.load(open(scrfd_json, "r"))

flag_head = False
with open(merge_bbox_file_path, "w") as f, open(os.path.join('datasets/tools/anno_json', "300wlp_box.json"), "w") as f_box, open(os.path.join('datasets/tools/anno_json', "300wlp_all.json"), "w") as f_all:
    f.write(f"{len(scrfd_json_data)}\n")
    f.write("image_id x_1 y_1 width height\n")

    for scrfd_data in scrfd_json_data:

        if not flag_head:
            f_box.write("[")
            f_all.write("[")
            flag_head = True
        else:
            f_box.write(",\n")
            f_all.write(",\n")

        img_path = os.path.join(folder_path,  scrfd_data["path"])
        mat_path = img_path[:-4] + ".mat"

        mat = io.loadmat(mat_path)
        pre_pose_params = mat['Pose_Para'][0]
        # Get [pitch, yaw, roll], And convert to degrees.
        pose_params = pre_pose_params[:3]
        pitch = round(pose_params[0] * 180 / np.pi, 4)
        yaw = round(pose_params[1] * 180 / np.pi, 4)
        roll = round(pose_params[2] * 180 / np.pi, 4)
        print(f"{img_path} pitch: {pitch}, yaw: {yaw}, roll: {roll}")
        
        pt2d = mat['pt2d']
        landmarks = np.array(np.transpose(mat['pt2d'])).astype('float').reshape(-1, 2)
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        # 稍微做一下 box 放大
        y_min = max(0, y_min - 0.2 * (y_max - y_min))
        w = x_max - x_min
        h = y_max - y_min
        mat_box = [int(x_min), int(y_min), int(w), int(h)]
        landmark = [int(landmarks[36][0] + landmarks[39][0])//2, int(landmarks[36][1] + landmarks[39][1])//2, \
                    int(landmarks[42][0] + landmarks[45][0])//2, int(landmarks[42][1] + landmarks[45][1])//2, \
                    int(landmarks[30][0]), int(landmarks[30][1]), \
                    int(landmarks[48][0]), int(landmarks[48][1]), \
                    int(landmarks[54][0]), int(landmarks[54][1])]
        
        if len(scrfd_data['result']) == 0:
            print(f"{img_path} no face detected, use the original bbox")
            f.write(f"{scrfd_data['path']} {mat_box[0]} {mat_box[1]} {mat_box[2]} {mat_box[3]}\n")
            f_box.write(json.dumps({
                "file_name": scrfd_data["path"],
                "bbox": mat_box
            }))
            f_all.write(json.dumps({
                "file_name": scrfd_data["path"],
                "bbox": mat_box,
                "headpose": [yaw, pitch, roll]
            }))
            continue

        # calculate the distance between the five landmark pairs
        mat_5_landmark = [[int(x), int(y)] for x,y in zip(landmark[0::2], landmark[1::2])]
        min_distance = 3000 # initialize the distance to a large number
        min_face_id = 'face_1'
        min_error_landmark = []
        for faces in scrfd_data['result']:
            for face_id, face_value in faces.items():
                face_landmark = face_value['landmarks']
                face_landmark_list = [face_landmark['left_eye'][0],face_landmark['left_eye'][1],face_landmark['right_eye'][0],face_landmark['right_eye'][1],face_landmark['nose'][0],face_landmark['nose'][1], \
                                    face_landmark['mouth_left'][0],face_landmark['mouth_left'][1],face_landmark['mouth_right'][0],face_landmark['mouth_right'][1]]

                face_landmark_list = [[int(float(x)), int(float(y))] for x,y in zip(face_landmark_list[0::2], face_landmark_list[1::2])]

                distance = np.linalg.norm(np.array(mat_5_landmark) - np.array(face_landmark_list))
                if distance < min_distance:
                    min_distance = distance
                    min_face_id = face_id
                    min_error_landmark = face_landmark_list
        
        assert min_face_id in scrfd_data['result'][int(min_face_id.split('_')[1])-1], f"{min_face_id} not in {img_path}"
        scrfd_box = scrfd_data['result'][int(min_face_id.split('_')[1])-1][min_face_id]['facial_area']
        assert len(scrfd_box) == 4, f"{scrfd_box} not in the right format"
        scrfd_box = [int(float(x)) for x in scrfd_box]

        ## check mat landmarks in scrfd box, at least four points in the box
        point_flag = 0
        for x,y in mat_5_landmark:
            if scrfd_box[0] < x < scrfd_box[0]+scrfd_box[2] and scrfd_box[1] < y < scrfd_box[1]+scrfd_box[3]:
                point_flag += 1    

        final_box = scrfd_box
        if point_flag < 4:
            # img = cv2.imread(os.path.join(folder_path,img_path))
            # for (x,y), (x2,y2) in zip(mat_5_landmark, min_error_landmark):
            #     cv2.circle(img, (x,y), 2, (0,255,0), 2) # 300W label, green
            #     cv2.circle(img, (x2,y2), 2, (255,0,0), 2) # scrfd result, blue
            # cv2.rectangle(img, (mat_box[0], mat_box[1]), (mat_box[0]+mat_box[2], mat_box[1]+mat_box[3]), (0,255,0), 2) # 300w label
            # cv2.rectangle(img, (scrfd_box[0], scrfd_box[1]), (scrfd_box[0]+scrfd_box[2], scrfd_box[1]+scrfd_box[3]), (255,0,0), 2) # scrfd result
            # os.makedirs(f"vis_samples/{os.path.dirname(scrfd_data['path'])}", exist_ok=True)
            # cv2.imwrite(f"vis_samples/{scrfd_data['path']}", img)
            
            # 如果关键点溢出了scrfd的box，使用300w的box
            print(f"{img_path} landmarks out of scrfd box, use the original bbox")
            final_box = mat_box

        f.write(f"{scrfd_data['path']} {final_box[0]} {final_box[1]} {final_box[2]} {final_box[3]}\n")



        box_dict = {
            "file_name": scrfd_data["path"],
            "bbox": final_box 
        }

        all_dict = box_dict.copy()
        all_dict.update({
            "headpose": [yaw, pitch, roll]
        })

        f_box.write(json.dumps(box_dict))
        f_all.write(json.dumps(all_dict))

    f_box.write("]")
    f_all.write("]")


