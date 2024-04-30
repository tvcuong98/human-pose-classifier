from utils import load_X, load_Y,norm_X
import numpy as np
from torch.utils.data import Dataset
import torch
# class Dataset_ske(Dataset):
#     def __init__(self, csv_path, transform=None, target_transform=None):
#         self.features = load_X(csv_path) # numpy , (num_graphs,num_nodes)
#         self.labels = load_Y(csv_path) # numpy , (num_graphs,)
#         self.features = norm_X(self.features)
#         # turn it into (num_graphs,num_feature,sequence_length,num_node) or (N,C,T,V) , but with C = 2 , T = 1 , V = 14
#         temp_features = np.zeros((self.features.shape[0],2,self.features.shape[1]//2)) # (num_graphs,2,num_node/2) with num_node=28
#         temp_features[:,0,:] = self.features[:,::2]
#         temp_features[:,1,:] = self.features[:,1::2]
#         self.features = temp_features
#         print(self.features.shape)
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         feature = self.features[idx]
#         label = self.labels[idx]
#         feature = torch.tensor(feature).unsqueeze(-2) # before any unsqueeze , it is (2,14)
#                                                                    # after unsqueeze two times , it is (1,2,1,14)
#         label = torch.tensor(label)
#         if self.transform:
#             feature = self.transform(feature)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return feature, label
class Dataset_ske(Dataset):
    # transform=None or transform = "norm_X"
    def __init__(self, csv_path, transform="norm_X"):
        self.features = load_X(csv_path) # numpy , (num_graphs,num_nodes)
        self.labels = load_Y(csv_path) # numpy , (num_graphs,)
        if transform=="norm_X":
            self.features=norm_X(self.features)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature = torch.tensor(feature)                                      
        label = torch.tensor(label)
        return feature, label
class Robust_Dataset_ske(Dataset):
    # 
    def __init__(self, csv_path, num_node=16,transform="norm_X"):
        tmp_features = load_X(csv_path) # numpy , (num_graphs,num_nodes) #m,28
        self.labels = load_Y(csv_path) # numpy , (num_graphs,)
        if transform=="norm_X":
            tmp_features = norm_X(tmp_features) #m,28
        self.features = np.zeros((tmp_features.shape[0],2,num_node))
        self.features[:,0,:] = tmp_features[:,0::2]
        self.features[:,1,:] = tmp_features[:,1::2]
        self.features=np.expand_dims(self.features,axis=2)
        self.transform = transform
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature = torch.tensor(feature)                             
        label = torch.tensor(label)
        return feature, label
