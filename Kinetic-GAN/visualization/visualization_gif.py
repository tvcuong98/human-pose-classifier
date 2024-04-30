import matplotlib.pyplot as plt
import os
import numpy as np
def plot_line(a, b):
    if (a.any()> 0 and b.any()>0): 
        plt.plot([a[0], b[0]], [a[1], b[1]], 'k-')

def plot_skeleton(sample, pattern):
    for i in range(len(sample)//2):
        plt.plot(sample[i*2], sample[i*2+1], pattern)
        plt.text(sample[i*2], sample[i*2+1], str(i), fontsize=8, ha='center', va='center')
    skeleton = sample.reshape(1, 32)
    Nose = skeleton[:,9*2:9*2+2][0]
    Neck = skeleton[:,8*2:8*2+2][0]
    Bottom = skeleton[:,0*2:0*2+2][0]
    Spine = skeleton[:,7*2:7*2+2][0]
    RWrist= skeleton[:,12*2:12*2+2][0]
    RElbow = skeleton[:,11*2:11*2+2][0]
    RShoulder = skeleton[:,10*2:10*2+2][0]
    LShoulder = skeleton[:,13*2:13*2+2][0]
    LElbow = skeleton[:,14*2:14*2+2][0]
    LWrist = skeleton[:,15*2:15*2+2][0]
    RAnkle = skeleton[:,6*2:6*2+2][0]
    RKnee = skeleton[:,5*2:5*2+2][0]
    RHip = skeleton[:,4*2:4*2+2][0]
    LHip = skeleton[:,1*2:1*2+2][0]
    LKnee = skeleton[:,2*2:2*2+2][0]
    LAnkle = skeleton[:,3*2:3*2+2][0]
    REye = skeleton[:,9*2:9*2+2][0]
    LEye = skeleton[:,9*2:9*2+2][0]
    REar = skeleton[:,9*2:9*2+2][0]
    LEar = skeleton[:,9*2:9*2+2][0]


    plot_line(Spine,Neck)
    plot_line(Bottom,Spine)
    plot_line(Bottom,LHip)
    plot_line(Bottom,RHip)


    plot_line(LEar, LEye)
    plot_line(LEye, Nose)
    plot_line(REar, REye)
    plot_line(REye, Nose)
    plot_line(Nose, Neck)
    plot_line(Neck, LShoulder)
    plot_line(LShoulder, LElbow)
    plot_line(LElbow, LWrist)
    plot_line(Neck, RShoulder)
    plot_line(RShoulder, RElbow)
    plot_line(RElbow, RWrist)
    #plot_line(Neck, LHip)
    plot_line(LHip, LKnee)
    plot_line(LKnee, LAnkle)
    #plot_line(Neck, RHip)
    plot_line(RHip, RKnee)
    plot_line(RKnee, RAnkle)

import torch
import matplotlib.pyplot as plt
import imageio
import io


def plot_gif_in_memory(tensor, filename):
    """
    Plots points from the given tensor and saves the plot as a GIF file,
    without saving intermediate frames to disk.

    Parameters:
        tensor (torch.Tensor): Input tensor of shape 2x32x16.
        filename (str): Name of the GIF file to save the animation. 
    """

    frames = []

    for frame_idx in range(tensor.shape[1]):
        frame_tensor = tensor[:, frame_idx, :]
        x_coordinates = frame_tensor[0] #16,
        y_coordinates = frame_tensor[1] #16,
        sample_in_1_frame = []
        for idx in range(frame_tensor.shape[1]):
            sample_in_1_frame.append(x_coordinates[idx])
            sample_in_1_frame.append(y_coordinates[idx])
        sample_in_1_frame = np.array(sample_in_1_frame)
        plot_skeleton(sample_in_1_frame, 'bo') # shape 32,
        plt.xlabel('X Coordinates')
        plt.ylabel('Y Coordinates')
        plt.title(f'Frame {frame_idx}')


        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.tight_layout()

        # Convert plot to image data (in memory)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        frames.append(imageio.imread(buffer))

        plt.close()

    # Save animation as GIF
    imageio.mimsave(filename, frames, duration=0.5)