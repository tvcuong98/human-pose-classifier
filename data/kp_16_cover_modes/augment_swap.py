import numpy as np
import csv
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()
parser.add_argument("--max_samples", type=int, default=48800, help="max_samples of each class after augmenting")
parser.add_argument("--input_csv", type=str, help="path to input csv")
parser.add_argument("--augmented_csv", type=str, help="path to output augmented csv")
opt = parser.parse_args()
print(opt)

def load_X(X_path):
    file = open(X_path, 'r')
    X_ = np.array(
        [elem for elem in [
            row.split(',')[:] for row in file
        ]],
        dtype=np.float32
    )
    X_=X_[:,:-1]
    print("X_.shape is ",str(X_.shape))
    file.close()
    return X_
def load_Y(Y_path):
    file = open(Y_path, 'r')
    Y_ = np.array(
        [elem for elem in [
            row.split(',')[:] for row in file
        ]],
        dtype=np.float32
    )
    Y_=Y_[:,-1]
    print("Y_shape is" ,str(Y_.shape))
    file.close()
    return Y_
def load_filepath(path):
    file = open(path, 'r')
    fp_=[]
    for row in file:
      temp_fp_=row.split(',')[-1]
      fp_.append(temp_fp_)
    print("Length of file_path is: ",len(fp_))
    file.close()
    return fp_

# Swapping upper parts and lower parts
max_samples=48800
df=pd.read_csv(opt.input_csv, header=None)

augmented_df_list=[]
with open(opt.augmented_csv, 'w', newline='') as sample_chosen_out_csv_file:
    sample_chosen_out_writer = csv.writer(sample_chosen_out_csv_file)
    for pose in range(0,9): 
        num_sample=0
        print(pose) 
        rows_pose_mask=df.iloc[:,-1]==pose
        rows_pose=df[rows_pose_mask]
        # Shuffle the rows
        rows_pose = rows_pose.sample(frac=1,random_state=123).reset_index(drop=True)
        print(f"class {pose} originally have {len(rows_pose)} samples")
        for row_pose_index_i in range (0,len(rows_pose)):
            sample_chosen_out_writer.writerow(list(rows_pose.iloc[row_pose_index_i]))
            num_sample+=1
        for row_pose_index_i in range (0,len(rows_pose)-1):
            if num_sample >=max_samples:
                break
            for row_pose_index_j in range (row_pose_index_i+1,len(rows_pose)):
                # trao lan 1
                new_keypoint=[0.0]*33
                new_keypoint[0:16]=list(rows_pose.iloc[row_pose_index_i,:16])
                new_keypoint[16:]=list(rows_pose.iloc[row_pose_index_j,16:])
                # tinh lai toa do spine
                new_keypoint[7*2]=(new_keypoint[0*2]+new_keypoint[8*2])/2.0
                new_keypoint[7*2+1]=(new_keypoint[0*2+1]+new_keypoint[8*2+1])/2.0
                sample_chosen_out_writer.writerow(new_keypoint)
                
                # trao lan 2
                new_keypoint=[0.0]*33
                new_keypoint[0:16]=list(rows_pose.iloc[row_pose_index_j,:16])
                new_keypoint[16:]=list(rows_pose.iloc[row_pose_index_i,16:])
                # tinh lai toa do spine
                new_keypoint[7*2]=(new_keypoint[0*2]+new_keypoint[8*2])/2.0
                new_keypoint[7*2+1]=(new_keypoint[0*2+1]+new_keypoint[8*2+1])/2.0
                sample_chosen_out_writer.writerow(new_keypoint)
                

                num_sample+=2
                if num_sample >=max_samples:
                    break
        print(f"class {pose} now have {num_sample}/{max_samples} samples. It is done")