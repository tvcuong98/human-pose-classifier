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
from classi_models.models import HeavyPoseClassifier
from models.PoseClassifier import RobustPoseClassifier
from utils.utils import norm_X
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_type",type=str,help="HeavyPoseClassifer,RobustPoseClassifier")
parser.add_argument("--classimodel", type=str, default="/ske/classifier/output/heavy_pose_classifier/good_checkpoint/checkpoint_normX_ep85_iterloss0.036_valloss0.266_valacc0.947.pt", help="path to classi model")
parser.add_argument("--classes", type=list, default=[0,1,2,3,4,5,6,7,8], help="classes that you want to sample")
parser.add_argument("--mode", type=str, help="pure,soft,hard")
parser.add_argument("--generator", type=str, default="/ske/Kinetic-GAN/runs-small/kinetic-gan/models/generator_300900.pth",help="path to generator weight")
parser.add_argument("--anglefilter", type=bool,help="do you want to use angle filter or not")
parser.add_argument("--maxsample", type=int, default=50000, help="max sample for all classes")
parser.add_argument("--listsample",type=list,help="number of samples for each of class")
parser.add_argument("--batchsize",type=int,default=1024,help=" batch size : how many samples should we generate each time")
parser.add_argument("--num_vis",type=int,default=20,help="at the end, how many samples should be visualize for each class")
parser.add_argument("--output",type=str,default="output")
opt = parser.parse_args()
print(opt)
cuda = True if torch.cuda.is_available() else False
print("Cuda is:",cuda)
import torch
# Define your custom operation
def evaluation_GAN_sample_soft(opt,cuda_batch_tensor,batch_label,classi_model):
    # use opt.model_type only
    model_type= opt.model_type
    if model_type in ["RobustPoseClassifier"]:
        # the cuda_batch_tensor is now #([batch_size, 32]) , and we need it to be #([batch_size, 2,1,16])(m,c,t,v)
        temp_cuda_batch_tensor = torch.zeros((cuda_batch_tensor.shape[0],2,16))
        temp_cuda_batch_tensor[:,0,:] = cuda_batch_tensor[:,0::2]
        temp_cuda_batch_tensor[:,1,:] = cuda_batch_tensor[:,1::2]
        temp_cuda_batch_tensor=torch.unsqueeze(temp_cuda_batch_tensor,dim=2)
        cuda_batch_tensor=temp_cuda_batch_tensor
    classes_groups=[[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # Output of the classifier model
    output = classi_model(cuda_batch_tensor)

    # Get the predicted classes
    _, predicted = torch.max(output.data, 1)

    # Initialize an empty list to store matching indices
    matching_indices = []

    # Iterate over each predicted class and batch label
    for pred, label in zip(predicted, batch_label):
        # Check if the predicted class and batch label belong to the same group
        for group in classes_groups:
            if pred.item() in group and label.item() in group:
                # If they belong to the same group, append the index to matching_indices
                matching_indices.append(True)
                break
        else:
            # If the predicted class and batch label do not belong to the same group, append False
            matching_indices.append(False)

    # Convert the matching indices to a NumPy array
    np_matching_indices = np.array(matching_indices)

    # Get the indices where the matching_indices array is True
    np_matching_indices = np_matching_indices.nonzero()[0]
    return np_matching_indices
def evaluation_GAN_sample_hard(cuda_batch_tensor,batch_label,classi_model):
    assert len(cuda_batch_tensor)==len(batch_label)
    model_type= opt.model_type
    if model_type in ["RobustPoseClassifier"]:
        # the cuda_batch_tensor is now #([batch_size, 32]) , and we need it to be #([batch_size, 2,1,16])(m,c,t,v)
        temp_cuda_batch_tensor = torch.zeros((cuda_batch_tensor.shape[0],2,16))
        temp_cuda_batch_tensor[:,0,:] = cuda_batch_tensor[:,0::2]
        temp_cuda_batch_tensor[:,1,:] = cuda_batch_tensor[:,1::2]
        temp_cuda_batch_tensor=torch.unsqueeze(temp_cuda_batch_tensor,dim=2)
        cuda_batch_tensor=temp_cuda_batch_tensor
    if cuda: cuda_batch_tensor=cuda_batch_tensor.cuda()
    output=classi_model(cuda_batch_tensor)
    _, predicted = torch.max(output.data, 1)
    matching_indices=(predicted==batch_label).nonzero(as_tuple=True)[0] # (5,) True,True,False,True,False because batch_size=5
    np_matching_indices=matching_indices.cpu().numpy()
    return np_matching_indices
def scaling_x_coord(x):
    # min_val = torch.tensor(0.0)
    # max_val = torch.tensor(120.0)
    min_val = torch.tensor(0.0)
    max_val = torch.tensor(255.0)
    scaled_x = (x + 1) / 2 * (max_val - min_val) + min_val
    return scaled_x
def scaling_y_coord(x):
    # min_val = torch.tensor(0.0)
    # max_val = torch.tensor(160.0)
    min_val = torch.tensor(0.0)
    max_val = torch.tensor(255.0)
    scaled_y = (x + 1) / 2 * (max_val - min_val) + min_val
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
if (opt.model_type in ["HeavyPoseClassifier"]):
    classi_model = HeavyPoseClassifier(input_size=32, num_classes=9, drop_out_p=0.0, hidden_dims=64)
elif (opt.model_type in ["RobustPoseClassifier"]):
    classi_model = RobustPoseClassifier(in_channels=2, n_classes=9, t_size=1, latent=512)

# model_state, optimizer_state = torch.load(os.path.join("runs/kinetic-gan/pretrained_models", "generator_bt300000_ep567.pth")) /home/edabk/Sleeping_pos/skeleton-based-code/pose_classifier/output/checkpoints/checkpoint_ep2.pt
#model_state,optimizer_state=torch.load("runs/kinetic-gan/classi_models/checkpoint_ep99_iterloss4.2033070069421984e-07_valloss0.5458492040634155_valacc0.9338970023059185.pt")
#model_state,optimizer_state=torch.load("/ske/classifier/output/heavy_pose_classifier/good_checkpoint/checkpoint_ep85_iterloss0.036683739152126936_valloss0.2669716477394104_valacc0.9476744186046512.pt")
model_state=torch.load(opt.classimodel)
classi_model.load_state_dict(model_state)
if cuda:classi_model.cuda()
classi_model.eval()
for name, param in classi_model.named_parameters():
    print(f"Parameter '{name}' has dtype: {param.dtype}")


list_classes=opt.classes
latent_dim=512
channels=2
n_classes=9
t_size=1
dataset='h36m'
mlp_dim=8
batch_size=opt.batchsize
if (opt.listsample==None):
    list_samples=[opt.maxsample]*9
else:
    list_samples=opt.listsample
GAN_model=opt.generator
# Initialize generator 
generator     = Generator(latent_dim, channels, n_classes, t_size, mlp_dim=mlp_dim,dataset=dataset)
cuda = True if torch.cuda.is_available() else False
print("Cuda is:",cuda)

if cuda:
    generator.cuda()


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Load Models
model,optim = torch.load(GAN_model)
generator.load_state_dict(model,strict=False)
# generator.load_state_dict(torch.load(GAN_model), strict=False)
generator.eval()
out         = "output"
gan_sample = os.path.join(out, opt.output)
if not os.path.exists(gan_sample): os.makedirs(gan_sample)
def sample_action(class_i,batch_size):
    batch_data_list=[]
    z = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim)))) # 10 *n_row means 10 samples per class # torch.Size([num_samples, 512])
    labels = np.array([class_i for _ in range(batch_size)])
    labels = Variable(LongTensor(labels))
    gen_frames = generator(z, labels).to(torch.device("cuda")) # torch.Size([batch_size, 2, 1, 16])
    gen_imgs = gen_frames[:, :, 0, :] # torch.Size([batch_size, 2, 16])
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
    matching_indexes_set_filter1=[] # this is for pure,soft,hard
    matching_indexes_set_filter2=[] # this is for angle filter
    if opt.mode=="soft":
        matching_indexes_list_soft=list(evaluation_GAN_sample_soft(torch_flatten,labels,classi_model))
        matching_indexes_set_soft = set(matching_indexes_list_soft)
        matching_indexes_set_filter1 = matching_indexes_set_soft
    elif opt.mode=="hard":
        matching_indexes_list_hard=list(evaluation_GAN_sample_hard(torch_flatten,labels,classi_model))
        matching_indexes_set_hard = set(matching_indexes_list_hard)
        matching_indexes_set_filter1 = matching_indexes_set_hard
    elif opt.mode=="pure":
        matching_indexes_set_filter1 = list(range(0, batch_size))

    if opt.anglefilter==True:
        matching_indexes_list_angle_filter=angle_filter(np_gen_imgs,class_i)
        matching_indexes_set_angle_filter = set(matching_indexes_list_angle_filter)
        matching_indexes_set_filter2 = matching_indexes_set_angle_filter
    elif opt.angle_filter==False:
        matching_indexes_set_filter2 = list(range(0, batch_size))
    
    # Find the common elements in both sets
    matching_indexes_combined = list(matching_indexes_set_filter1.intersection(matching_indexes_set_filter2))


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
    #for class_i in range(0,n_classes):
    for class_i in list_classes:
        num_samples=list_samples[int(class_i)]
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
                if count_sample%(num_samples//opt.num_vis)==(num_samples//opt.num_vis)-1:
                    additional_display_samples.plot(numpy_row,str("gen_class{}_keypoint{}.png").format(str(class_i),count_sample),os.path.join(gan_sample,str(class_i)))
            if (count_sample>=num_samples):
                print(f"Class {class_i} now have {count_sample}/{num_samples}")
                count_sample=0
                done_class_i=True

            

        