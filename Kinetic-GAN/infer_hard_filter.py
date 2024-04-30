import os
import numpy as np
import csv
import pandas as pd
import os
import torch
from torch.autograd import Variable
from models.generator import Generator
from visualization import additional_display_samples
import math
import sys
# sys.path.insert(0,'/ske/classifier')
# print(sys.path)
from classi_models.models import HeavyPoseClassifier
from utils.utils import norm_X
cuda = True if torch.cuda.is_available() else False
print("Cuda is:",cuda)
import torch
# Minimum x-coordinate: 15.59075043630018
# Maximum x-coordinate: 101.8734729493892
# Minimum y-coordinate: 0.7197309417040572
# Maximum y-coordinate: 146.53986429177272

# Define your custom operation
def scaling_x_coord(x):
    min_val = torch.tensor(0.0)
    max_val = torch.tensor(120.0)
    scaled_x = (x + 1) / 2 * (max_val - min_val) + min_val
    return scaled_x
def scaling_y_coord(y):
    min_val = torch.tensor(0.0)
    max_val = torch.tensor(160.0)
    scaled_y = (y + 1) / 2 * (max_val - min_val) + min_val
    return scaled_y
def adjust_bottom_spine(batch_tensor): #batch,2,16
    """
    Adjusts node coordinates based on specified rules.

    Args:
    - batch_tensor (torch.Tensor): Input tensor with shape (batch_size, 2, 16).

    Returns:
    - torch.Tensor: Tensor with adjusted node coordinates.
    """
    # Accessing nodes
    node_1 = batch_tensor[:, :, 1]
    node_4 = batch_tensor[:, :, 4]
    # Assigning new coordinates for node 0
    batch_tensor[:, :, 0] = (node_1 + node_4) / 2
    # accessing node 0,node 8
    node_8 = batch_tensor[:, :, 8]
    node_0 = batch_tensor[:, :, 0]
    # Assigning new coordinates for node 7
    batch_tensor[:, :, 7] = (node_0 + node_8) / 2
    return batch_tensor
def calculate_angle(joints, index_A, index_B, index_C): # joints is 32 elements list of x0,y0 ,.... x31,y31
    # Extracting coordinates of joints A, B, and C
    Ax, Ay = joints[2*index_A], joints[2*index_A+1]
    Bx, By = joints[2*index_B], joints[2*index_B+1]
    Cx, Cy = joints[2*index_C], joints[2*index_C+1]

    # Calculating vectors BA and BC
    BAx, BAy = Ax - Bx, Ay - By
    BCx, BCy = Cx - Bx, Cy - By

    # Calculating dot product and magnitudes
    dot_product = BAx * BCx + BAy * BCy
    magnitude_BA = math.sqrt(BAx**2 + BAy**2)
    magnitude_BC = math.sqrt(BCx**2 + BCy**2)

    # Calculating angle between BA and BC
    # print(f"dot_product {dot_product} magnitude_BA {magnitude_BA} magnitude_BC {magnitude_BC}")
    # print(dot_product / (magnitude_BA * magnitude_BC))
    # angle_radians = math.acos(dot_product / (magnitude_BA * magnitude_BC))
    angle_radians = math.acos(max(min(dot_product / (magnitude_BA * magnitude_BC), 1), -1))
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees 
def angle_filter(batch_numpy,class_i): #batch,2,16 and is numpy
    rules = [{"co":[160.0,180.0],"duoi":[160.0,180.0]}, #0
             {"co":[26.0,110.0],"duoi":[160.0,180.0]}, #1
             {"co":[0.0,120.0],"duoi":[0.0,120.0]}, #2
             {"co":[120.0,180.0],"duoi":[120.0,180.0]},  #3
             {"co":[30.0,100.0],"duoi":[120.0,180.0]}, #4
             {"co":[30.0,110.0],"duoi":[30.0,110.0]}, #5
             {"co":[125.0,180.0],"duoi":[125.0,180.0]}, #6
             {"co":[27.0,110.0],"duoi":[125.0,180.0]}, #7
             {"co":[26.0,110.0],"duoi":[26.0,110.0]}, #8
             ]
    assert class_i in [0,1,2,3,4,5,6,7,8]
    matching_indexes =[]
    for idx in range(len(batch_numpy)):#batch,2,16
        x_coords = batch_numpy[idx,0,:] 
        y_coords = batch_numpy[idx,1,:] 
        joints =[]
        for node_idx in range(batch_numpy.shape[-1]):
            joints.append(x_coords[node_idx])
            joints.append(y_coords[node_idx])
        angle_phai = calculate_angle(joints,4,5,6)
        angle_trai = calculate_angle(joints,1,2,3)
        max_angle = max(angle_phai, angle_trai)
        min_angle = min(angle_phai, angle_trai)
        if min_angle >= rules[class_i]["co"][0] and min_angle <= rules[class_i]["co"][1] and max_angle >= rules[class_i]["duoi"][0] and max_angle <= rules[class_i]["duoi"][1]:
            matching_indexes.append(idx)
    return matching_indexes








classi_model = HeavyPoseClassifier(input_size=32, num_classes=9, drop_out_p=0.0, hidden_dims=64)
if cuda:
    classi_model.cuda()
# model_state, optimizer_state = torch.load(os.path.join("runs/kinetic-gan/pretrained_models", "generator_bt300000_ep567.pth")) /home/edabk/Sleeping_pos/skeleton-based-code/pose_classifier/output/checkpoints/checkpoint_ep2.pt
#model_state,optimizer_state=torch.load("runs/kinetic-gan/classi_models/checkpoint_ep99_iterloss4.2033070069421984e-07_valloss0.5458492040634155_valacc0.9338970023059185.pt")
#model_state,optimizer_state=torch.load("/ske/classifier/output/heavy_pose_classifier/good_checkpoint/checkpoint_ep85_iterloss0.036683739152126936_valloss0.2669716477394104_valacc0.9476744186046512.pt")
model_state=torch.load("/ske/classifier/output/heavy_pose_classifier/good_checkpoint/checkpoint_normX_ep85_iterloss0.036_valloss0.266_valacc0.947.pt")
classi_model.load_state_dict(model_state)
if cuda:
    classi_model.cuda()
classi_model.eval()
for name, param in classi_model.named_parameters():
    print(f"Parameter '{name}' has dtype: {param.dtype}")



latent_dim=512
channels=2
n_classes=9
t_size=32
dataset='h36m'
mlp_dim=8
batch_size=512  
num_samples=2000
#GAN_model="/ske/Kinetic-GAN/runs/kinetic-gan/models/generator_145350.pth"
GAN_model="/ske/Kinetic-GAN/runs-small/kinetic-gan/models/generator_300900.pth"

# Initialize generator 
generator     = Generator(latent_dim, channels, n_classes, t_size, mlp_dim=mlp_dim,dataset=dataset)
cuda = True if torch.cuda.is_available() else False
print("Cuda is:",cuda)

if cuda:
    generator.cuda()
def evaluation_GAN_sample(cuda_batch_tensor,batch_label,classi_model):
    assert len(cuda_batch_tensor)==len(batch_label)
    output=classi_model(cuda_batch_tensor)
    _, predicted = torch.max(output.data, 1)
    matching_indices=(predicted==batch_label).nonzero(as_tuple=True)[0] # (5,) True,True,False,True,False because batch_size=5
    np_matching_indices=matching_indices.cpu().numpy()
    # print(np_matching_indices)
    return np_matching_indices
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Load Models
model,optim = torch.load(GAN_model)
generator.load_state_dict(model,strict=False)
# generator.load_state_dict(torch.load(GAN_model), strict=False)
generator.eval()
out         = "output"
gan_sample = os.path.join(out, 'gan_sample')
if not os.path.exists(gan_sample): os.makedirs(gan_sample)
def sample_action(class_i,batch_size):
    batch_data_list=[]
    z = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim)))) # 10 *n_row means 10 samples per class # torch.Size([num_samples, 512])
    labels = np.array([class_i for _ in range(batch_size)])
    labels = Variable(LongTensor(labels))
    gen_frames = generator(z, labels).to(torch.device("cuda")) # torch.Size([batch_size, 2, t_size, 16])
    # Choose a random number between 0 and shape[2]
    random_index = torch.randint(0, gen_frames.shape[2], (1,)).item()
    # Create a new tensor equal to gen_imgs[:, :, random_index, :]
    gen_imgs = gen_frames[:, :, random_index, :] # torch.Size([batch_size, 2, 16])
    gen_imgs = adjust_bottom_spine(gen_imgs)
    gen_imgs[:, 0, :] = scaling_x_coord(gen_imgs[:, 0, :]) # torch.Size([batch_size, 16])
    gen_imgs[:, 1, :] = scaling_y_coord(gen_imgs[:, 1, :]) # torch.Size([batch_size, 16])

    ## this part is separated, only for retrieving the filtered index that we are going to chooose as gan sample
    np_gen_imgs=gen_imgs.data.cpu().numpy() #([batch_size, 2, 16])
    np_x_coords = np_gen_imgs[:,0,:] #([batch_size, 16])
    np_y_coords = np_gen_imgs[:,1,:] #([batch_size, 16])
    np_flatten = np.zeros((np_gen_imgs.shape[0],2*np_gen_imgs.shape[-1])) #([batch_size, 32])
    for i in range(np_flatten.shape[-1]): # FROM 0 TO 32
        np_flatten[:,0::2] = np_x_coords
        np_flatten[:,1::2] = np_y_coords
    np_flatten = norm_X(np_flatten)
    if cuda: torch_flatten=torch.tensor(np_flatten).cuda() 
    else: torch_flatten=torch.tensor(np_flatten)
    # make it the same dtype with the weight of classi model
    torch_flatten=torch_flatten.to(torch.float32)
    matching_indexes=evaluation_GAN_sample(torch_flatten,labels,classi_model)
    matching_indexes_list1=list(matching_indexes)
    matching_indexes_set1 = set(matching_indexes_list1)
    ## now applying the second filter : 
    # Create np_gen_imgs with rows matching the indexes in matching_indexes
    matching_indexes_list2=angle_filter(np_gen_imgs,class_i)
    matching_indexes_set2 = set(matching_indexes_list2)

    # Find the common elements in both sets
    matching_indexes_combined = list(matching_indexes_set1.intersection(matching_indexes_set2))

    
    #print(matching_indexes_list)
    # print(evaluation_GAN_sample(norm_gen_imgs,classi_model=classi_model))
    for i in matching_indexes_combined:
        row_data_list=[]
        # data = gen_imgs.data[i] # it will be tensor + cuda
        # data = gen_imgs.data[i].cpu() # it will be tensor
        data = gen_imgs.data[i].cpu().numpy()# only now will it print out the numbers
        x_coords = data[0,:] #14
        y_coords = data[1,:] #14
        for i in range(data.shape[-1]):
            row_data_list.append(x_coords[i])
            row_data_list.append(y_coords[i])
        row_data_list.append(class_i)
        batch_data_list.append(row_data_list)
        # if not os.path.exists(os.path.join(sample_chosen_out,str(class_i))): os.makedirs(os.path.join(sample_chosen_out,str(class_i)))
        # additional_display_samples.plot(gen_imgs.data[i],str("gen_keypoint{}").format(i),os.path.join(sample_chosen_out,str(class_i)))
    return batch_data_list
with open(os.path.join(gan_sample, 'gan_sample.csv'), 'w', newline='') as gan_csv:
    # Create a CSV writer object
    gan_writer = csv.writer(gan_csv)
    # Write each line to the CSV file
    for class_i in range(0,n_classes):
        if not os.path.exists(os.path.join(gan_sample,str(class_i))): os.makedirs(os.path.join(gan_sample,str(class_i)))
        done_class_i=False
        count_sample=0
        while (done_class_i == False):
            batch_data_list=sample_action(class_i,batch_size)
            for row in batch_data_list:
                gan_writer.writerow(row) # row is a 33 element list
                count_sample+=1
                numpy_row = np.array(row[:-1]) #32,
                if count_sample%20==19:
                    print(f"Class {class_i} now have {count_sample}/{num_samples}")
                if count_sample%(num_samples//20)==(num_samples//20)-1:
                    additional_display_samples.plot(numpy_row,str("gen_class{}_keypoint{}.png").format(str(class_i),count_sample),os.path.join(gan_sample,str(class_i)))
            if (count_sample>=num_samples):
                print(f"Class {class_i} now have {count_sample}/{num_samples}")
                count_sample=0
                done_class_i=True

            

        