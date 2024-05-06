from data_utils import load_X, load_Y,norm_X
import numpy as np
from torch.utils.data import Dataset
import torch
class Ske_dataset(Dataset):
    # 
    def __init__(self, csv_path, num_node=16,transform="norm_X"):
        tmp_features = load_X(csv_path) # numpy , (num_graphs,num_nodes) #m,28
        self.labels = load_Y(csv_path) # numpy , (num_graphs,)
        if transform=="norm_X":
            tmp_features = norm_X(tmp_features) #m,32
        self.features = np.zeros((tmp_features.shape[0],2,num_node))
        self.features[:,0,:] = tmp_features[:,0::2]
        self.features[:,1,:] = tmp_features[:,1::2]
        # now it is N x 2 x 16
        self.features = self.features.transpose(0, 2, 1) 
        self.transform = transform
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature = torch.tensor(feature)                             
        label = torch.tensor(label)
        return feature, label