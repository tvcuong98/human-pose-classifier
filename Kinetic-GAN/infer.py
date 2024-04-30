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
cuda = True if torch.cuda.is_available() else False
print("Cuda is:",cuda)
import torch
# Minimum x-coordinate: 15.59075043630018
# Maximum x-coordinate: 101.8734729493892
# Minimum y-coordinate: 0.7197309417040572
# Maximum y-coordinate: 146.53986429177272

# Define your custom operation
def scaling_y_coord(x):
    min_val = torch.tensor(0.0)
    max_val = torch.tensor(120.0)
    scaled_x = (x + 1) / 2 * (max_val - min_val) + min_val
    return scaled_x
def scaling_x_coord(x):
    min_val = torch.tensor(0.0)
    max_val = torch.tensor(160.0)
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
latent_dim=512
channels=2
n_classes=9
t_size=32
dataset='h36m'
mlp_dim=8
batch_size=256  
num_samples=40000
#GAN_model="/ske/Kinetic-GAN/runs/kinetic-gan/models/generator_145350.pth"
# GAN_model="/ske/Kinetic-GAN/runs/kinetic-gan/models/generator_300900.pth"
GAN_model="/ske/Kinetic-GAN/runs-small/kinetic-gan/models/generator_51000.pth"
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
    # print(evaluation_GAN_sample(norm_gen_imgs,classi_model=classi_model))
    for i in range(batch_size):
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
                if count_sample%100==99:
                    print(f"Class {class_i} now have {count_sample}/{num_samples}")
                if count_sample%4000==3999:
                    additional_display_samples.plot(numpy_row,str("gen_class{}_keypoint{}.png").format(str(class_i),count_sample),os.path.join(gan_sample,str(class_i)))
            if (count_sample>=num_samples):
                print(f"Class {class_i} now have {count_sample}/{num_samples}")
                count_sample=0
                done_class_i=True

            

        