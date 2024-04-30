import os
from cv2 import norm
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import torch
# from utils import norm_X
import numpy as np
def euclidean_dist(a, b):
    # This function calculates the euclidean distance between 2 point in 2-D coordinates
    # if one of two points is (0,0), dist = 0
    # a, b: input array with dimension: m, 2
    # m: number of samples
    # 2: x and y coordinate
    # print(a.shape)
    # print(b.shape)
    try:
        if (a.shape[1] == 2 and a.shape == b.shape):
            # check if element of a and b is (0,0)
            bol_a = (a[:,0] != 0).astype(int)
            bol_b = (b[:,0] != 0).astype(int)
            dist = np.linalg.norm(a-b, axis=1)
            # print("haha")
            return((dist*bol_a*bol_b).reshape(a.shape[0],1))
    except:
        print("[Error]: Check dimension of input vector")
        return 0
def norm_X(X):
    num_sample = X.shape[0]
    # Keypoints
    # print(X[0,:])
    Nose = X[:,9*2:9*2+2]
    Neck = X[:,8*2:8*2+2]
    RShoulder = X[:,10*2:10*2+2]
    RElbow = X[:,11*2:11*2+2]
    RWrist = X[:,12*2:12*2+2]
    LShoulder = X[:,13*2:13*2+2]
    LElbow = X[:,14*2:14*2+2]
    LWrist = X[:,15*2:15*2+2]
    RHip = X[:,4*2:4*2+2]
    RKnee = X[:,5*2:5*2+2]
    RAnkle = X[:,6*2:6*2+2]
    LHip = X[:,1*2:1*2+2]
    LKnee = X[:,2*2:2*2+2]
    LAnkle = X[:,3*2:3*2+2]
    # REye = X[:,14*2:14*2+2]
    # LEye = X[:,15*2:15*2+2]
    # REar = X[:,16*2:16*2+2]
    # LEar = X[:,17*2:17*2+2]
    REye = X[:,9*2:9*2+2]
    LEye = X[:,9*2:9*2+2]
    REar = X[:,9*2:9*2+2]
    LEar = X[:,9*2:9*2+2]

    # Length of head
    length_Neck_LEar = euclidean_dist(Neck, LEar)
    length_Neck_REar = euclidean_dist(Neck, REar)
    length_Neck_LEye = euclidean_dist(Neck, LEye)
    length_Neck_REye = euclidean_dist(Neck, REye)
    length_Nose_LEar = euclidean_dist(Nose, LEar)
    length_Nose_REar = euclidean_dist(Nose, REar)
    length_Nose_LEye = euclidean_dist(Nose, LEye)
    length_Nose_REye = euclidean_dist(Nose, REye)
    length_head      = np.maximum.reduce([length_Neck_LEar, length_Neck_REar, length_Neck_LEye, length_Neck_REye, \
                                 length_Nose_LEar, length_Nose_REar, length_Nose_LEye, length_Nose_REye])
    #length_head      = np.sqrt(np.square((LEye[:,0:1]+REye[:,0:1])/2 - Neck[:,0:1]) + np.square((LEye[:,1:2]+REye[:,1:2])/2 - Neck[:,1:2]))

    # Length of torso
    length_Neck_LHip = euclidean_dist(Neck, LHip)
    length_Neck_RHip = euclidean_dist(Neck, RHip)
    length_torso     = np.maximum(length_Neck_LHip, length_Neck_RHip)
    #length_torso     = np.sqrt(np.square(Neck[:,0:1]-(LHip[:,0:1]+RHip[:,0:1])/2) + np.square(Neck[:,1:2]-(LHip[:,1:2]+RHip[:,1:2])/2))

    # Length of right leg
    length_leg_right = euclidean_dist(RHip, RKnee) + euclidean_dist(RKnee, RAnkle)
    #length_leg_right = np.sqrt(np.square(RHip[:,0:1]-RKnee[:,0:1]) + np.square(RHip[:,1:2]-RKnee[:,1:2])) \
    #+ np.sqrt(np.square(RKnee[:,0:1]-RAnkle[:,0:1]) + np.square(RKnee[:,1:2]-RAnkle[:,1:2]))

    # Length of left leg
    length_leg_left = euclidean_dist(LHip, LKnee) + euclidean_dist(LKnee, LAnkle)
    #length_leg_left = np.sqrt(np.square(LHip[:,0:1]-LKnee[:,0:1]) + np.square(LHip[:,1:2]-LKnee[:,1:2])) \
    #+ np.sqrt(np.square(LKnee[:,0:1]-LAnkle[:,0:1]) + np.square(LKnee[:,1:2]-LAnkle[:,1:2]))

    # Length of leg
    length_leg = np.maximum(length_leg_right, length_leg_left)

    # Length of body
    length_body = length_head + length_torso + length_leg

    # Check all samples have length_body of 0
    length_chk = (length_body > 0).astype(int)

    # Check keypoints at origin
    keypoints_chk = (X > 0).astype(int)

    chk = length_chk * keypoints_chk

    # Set all length_body of 0 to 1 (to avoid division by 0)
    length_body[length_body == 0] = 1

    # The center of gravity
    # number of point OpenPose locates:
    # X.shape = (None,m,36) with 36 = 18 nodes * 2 and m samples
    # X[:, 0::2].shape = (None,m,18)
    # (X[:, 0::2] > 0).shape = (None,m,18) , but for each of the m rows , for each of the element in the row:
    # [1,2.2,3.1,0,2] -> [True,True,True,False,True]
    # [1,0,0,4,2]     -> [True,False,False,True,True]
    # Similarly for m rows
    num_pts = (X[:, 0::2] > 0).sum(1).reshape(num_sample,1) #(None,m,1) where each row represent the number of valid nodes that have x_coord > 0
    centr_x = X[:, 0::2].sum(1).reshape(num_sample,1) / num_pts
    centr_y = X[:, 1::2].sum(1).reshape(num_sample,1) / num_pts

    # The  coordinates  are  normalized relative to the length of the body and the center of gravity
    X_norm_x = (X[:, 0::2] - centr_x) / length_body
    X_norm_y = (X[:, 1::2] - centr_y) / length_body

    # Stack 1st element x and y together
    X_norm = np.column_stack((X_norm_x[:,:1], X_norm_y[:,:1]))

    for i in range(1, X.shape[1]//2):
        X_norm = np.column_stack((X_norm, X_norm_x[:,i:i+1], X_norm_y[:,i:i+1]))

    # Set all samples have length_body of 0 to origin (0, 0)
    X_norm = X_norm * chk

    return X_norm
def load_X(X_path):
    file = open(X_path, 'r')
    X_ = np.array(
        [np.array(elem,dtype=np.float32) for elem in [
            # row.split(',')[:-1] for row in file # uncomment this if you are using tensor_data_train_9poses_classifier.csv
            row.split(',')[:] for row in file # uncooment this if using tensor_data_train_9poses_classifier.csv
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
        [np.array(elem,dtype=np.float32) for elem in [
            # row.split(',')[:-1] for row in file # uncomment this if you are using tensor_data_train_9poses_classifier.csv
            row.split(',')[:] for row in file # uncooment this if using tensor_data_train_9poses_classifier.csv
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
class Feeder(Dataset):
    def __init__(self, csv_path,v_size,t_size,transform=None, target_transform=None):
        self.v_size=v_size
        self.t_size=t_size
        self.features = load_X(csv_path) # numpy , (num_graphs,num_nodes)
        self.labels = load_Y(csv_path) # numpy , (num_graphs,)
        self.features=norm_X(self.features)
        # turn it into (num_graphs,num_feature,sequence_length,num_node) or (N,C,T,V) , but with C = 2 , T = 1 , V = 14
        temp_features = np.zeros((self.features.shape[0],2,self.features.shape[1]//2)) # (num_graphs,2,num_node/2) with num_node=28
        temp_features[:,0,:] = self.features[:,::2]
        temp_features[:,1,:] = self.features[:,1::2]
        self.features = temp_features
        self.max, self.min = self.features.max(), self.features.min()
        #print(self.features.shape)
        self.transform = transform
        self.target_transform = target_transform
    def class_wise_std(self,standardized):
        assert isinstance(standardized, (str, )), "standardized must a mode, between [mean, xmax_involved, xmax_xmin_involved]"

        # do it in here
        class_stds = []
        for class_label in np.unique(self.labels): # np.unique(self.labels) = [0. 1. 2. 3. 4. 5. 6. 7. 8.]
            # Get data samples belonging to the current class
            class_data = self.features[self.labels == class_label]

            # Calculate the standard deviation along the samples axis (axis=0) for each node, accross all the samples in the same class
            std_per_node = np.std(class_data, axis=0) # shape (2,16)
            if (standardized == "mean"): 
                standardized_std =  np.mean(std_per_node) # this is a scalar mean
                class_stds.append(standardized_std)
            elif (standardized == "xmax_involved"): 
                sum_std_all_nodes =  np.sum(std_per_node) 
                class_stds.append(sum_std_all_nodes)
                class_stds = [class_std/max(class_stds) for class_std in class_stds]
            elif (standardized == "xmax_xmin_involved"): 
                sum_std_all_nodes =  np.sum(std_per_node) 
                class_stds.append(sum_std_all_nodes)
                class_stds = [(class_std-min(class_stds))/max(class_stds)-min(class_stds) for class_std in class_stds]

        return np.array(class_stds) # (9,) since 9 classes
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature = torch.tensor(feature).unsqueeze(-2) # before any unsqueeze , it is (2,16)
                                                                   # after unsqueeze one times , it is (2,1,16)
        feature = 2 * ((feature - self.min) / (self.max - self.min)) - 1
        feature = feature.expand(feature.shape[0],self.t_size,self.v_size) #2,32,16
        label = torch.tensor(label)
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)
        return feature, label