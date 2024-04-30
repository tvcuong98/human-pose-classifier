import cv2  # Import OpenCV for video/image processing
from ultralytics import YOLO
import numpy as np
import imageio
import csv
import time
import os
# os.environ["YOLO_VERBOSE"] = "False" 
directory = "path_to_image_directory"
#model_path = "yolov8l-pose.pt"
model_path = "yolov8x-pose-p6.pt"
output_path_norm = "heavy/train_keypoint_norm"
output_path_base = "heavy/train_keypoint_base"
# Load YOLOv8 model
model = YOLO(model_path)
# # Export the model to ONNX format
# model.export(format='onnx',imgsz=[1280,1280])  # creates 'yolov8n.onnx'
# # Load the exported ONNX model
# model = YOLO('yolov8x-pose-p6.onnx')
# print(model.__dir__())
# Define path to video file
def video_bbox_keypoints(video_path,model):
    bboxes_n = []
    skeletons_n = []
    bboxes = []
    skeletons = []

    # Video and model paths



    source = video_path

    # results = model.predict(source=source, stream=True,conf=0.4)
    results = model.predict(source=source, stream=True,verbose=False)
    for result in results:
        # check if any pose and box were detected, if not then skip this frame
        if (len(result.boxes.xyxyn)<1 or len(result.keypoints.xyn)<1):
            continue
        if (len(result.boxes.xyxy)<1 or len(result.keypoints.xy)<1):
            continue
        # Detection normalize
        bbox_n = result.boxes.xyxyn[0].cpu().numpy() # shape (4,) , numpy array
        keypoints_n = result.keypoints.xyn[0].cpu().numpy() # shape (17,2), numpy array
        bboxes_n.append([*bbox_n]) # take the elements of bbox
        skeletons_n.append(keypoints_n)

        # Detection usual
        bbox = result.boxes.xyxy[0].cpu().numpy() # shape (4,) , numpy array
        keypoints = result.keypoints.xy[0].cpu().numpy() # shape (17,2), numpy array
        bboxes.append([*bbox]) # take the elements of bbox
        skeletons.append(keypoints)

    return bboxes,skeletons,bboxes_n,skeletons_n

    # Each result is composed of torch.Tensor by default,
    # in which you can easily use following functionality:
    # result = result.cuda()
    # result = result.cpu()
    # result = result.to("cpu")
    # result = result.numpy()
import os

total_files = len(os.listdir(directory))
files_processed = 0
if not os.path.exists(output_path_norm): os.makedirs(output_path_norm)
if not os.path.exists(output_path_base): os.makedirs(output_path_base)
for file in os.listdir(directory):
    input_video_path = os.path.join(directory,file)
    if os.path.isfile(input_video_path):
        start_time = time.time()  # Time the processing
        # Do something with each file
        base_name = os.path.splitext(os.path.basename(input_video_path))[0]
        new_file_name = base_name + ".csv"
        output_file_norm = os.path.join(output_path_norm,new_file_name)
        output_file_base = os.path.join(output_path_base,new_file_name)
        with open(output_file_norm ,"w",newline="") as file_norm, open(output_file_base ,"w",newline="") as file_base:
            writer_norm = csv.writer(file_norm)
            writer_base = csv.writer(file_base)
            bboxes,skeletons,bboxes_n,skeletons_n = video_bbox_keypoints(video_path=input_video_path,model=model)
            assert len(bboxes) == len(skeletons)
            assert len(bboxes_n) == len(skeletons_n)
            for i in range(len(bboxes)):
                row_data = [0] * 38
                bbox = bboxes[i]
                skeleton = skeletons[i]
                # for the bbox:
                row_data[34:] = bbox[0:3]
                # for the skeleton
                row_data[0:33:2] = skeleton[:,0]
                row_data[1:34:2] = skeleton[:,1]
                writer_base.writerow(row_data)
                ####################################### normalized data
                row_data_norm = [0] * 38
                bbox_n = bboxes_n[i]
                skeleton_n = skeletons_n[i]
                # for the bbox:
                row_data_norm[34:] = bbox_n[0:3]
                # for the skeleton
                row_data_norm[0:33:2] = skeleton_n[:,0]
                row_data_norm[1:34:2] = skeleton_n[:,1]
                writer_norm.writerow(row_data_norm)
        files_processed += 1
        process_time = time.time() - start_time

        print(f"Done processing: {input_video_path}")
        print(f"Files processed: {files_processed}/{total_files} in {process_time:.2f} sec")
