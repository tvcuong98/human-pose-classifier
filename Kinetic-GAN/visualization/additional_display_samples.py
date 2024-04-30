import numpy as np
from matplotlib import pyplot as plt
import os
def plot_line(a, b):
    if (a.any()> 0 and b.any()>0): plt.plot([a[0], b[0]], [a[1], b[1]], 'k-')

def plot_skeleton(sample, pattern):
    # for i in range(len(sample)//2):
    #     plt.plot(sample[i*2], sample[i*2+1], pattern)
    skeleton = sample.reshape(1, 32)
    Nose = skeleton[:,9*2:9*2+2][0]
    Neck = skeleton[:,8*2:8*2+2][0]
    Spine = skeleton[:,7*2:7*2+2][0]
    Bottom = skeleton[:,0*2:0*2+2][0]
    RShoulder = skeleton[:,10*2:10*2+2][0]
    RElbow = skeleton[:,11*2:11*2+2][0]
    RWrist = skeleton[:,12*2:12*2+2][0]
    LShoulder = skeleton[:,13*2:13*2+2][0]
    LElbow = skeleton[:,14*2:14*2+2][0]
    LWrist = skeleton[:,15*2:15*2+2][0]
    RHip = skeleton[:,4*2:4*2+2][0]
    RKnee = skeleton[:,5*2:5*2+2][0]
    RAnkle = skeleton[:,6*2:6*2+2][0]
    LHip = skeleton[:,1*2:1*2+2][0]
    LKnee = skeleton[:,2*2:2*2+2][0]
    LAnkle = skeleton[:,3*2:3*2+2][0]
    # REye = skeleton[:,14*2:14*2+2][0]
    # LEye = skeleton[:,15*2:15*2+2][0]
    # REar = skeleton[:,16*2:16*2+2][0]
    # LEar = skeleton[:,17*2:17*2+2][0]
    REye = skeleton[:,9*2:9*2+2][0]
    LEye = skeleton[:,9*2:9*2+2][0]
    REar = skeleton[:,9*2:9*2+2][0]
    LEar = skeleton[:,9*2:9*2+2][0]
    #Nose = sample.reshape(1, 36)[:,0*2:0*2+2][0]
    #Neck = sample.reshape(1, 36)[:,1*2:1*2+2][0]

    x_coordinates=skeleton[:,0::2][0]
    y_coordinates=skeleton[:,1::2][0]
    #Annotate each point with its index along the last dimension of the tensor
    for i, (x, y) in enumerate(zip(x_coordinates, y_coordinates)):
        plt.text(x, y, str(i), fontsize=10, ha='center', va='center')
    plot_line(Neck,Spine)
    plot_line(Spine,Bottom)
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
    plot_line(LHip, LKnee)
    plot_line(LKnee, LAnkle)
    plot_line(RHip, RKnee)
    plot_line(RKnee, RAnkle)

def plot(sample,name,path): # this is for ploting 1 sample
    # e.g the real shape is: (1,32) ( 2 is x and y)  (coi nhu ta ko biet 32 di, cu coi nhu la sample.shape[1])
    # temp_sample=np.zeros((32,))
    # temp_sample[::2]=sample[0,:]
    # temp_sample[1::2]=sample[1,:]
    # sample=temp_sample


    if sample.shape[0] == 32:
        sample_norm = sample.reshape(1,32)[0]

        # Plot original coordinates
        pad_ori = 40
        plt.figure(str(sample))
        plt.subplot(121)
        plt.title('Original skeleton')
        X_ori = sample
        x_max = max(X_ori[0::2]) + pad_ori
        x_min = min(i for i in X_ori[0::2] if i > 0) - pad_ori
        y_max = max(X_ori[1::2]) + pad_ori
        y_min = min(j for j in X_ori[1::2] if j > 0) - pad_ori
        plt.xlim(x_min,x_max)
        plt.ylim(y_max, y_min)
        plot_skeleton(X_ori, 'bo')

        # Plot normalized coordinates
        pad_nor = 0.2
        #plt.figure(2)
        plt.subplot(122)
        plt.title('Normalized skeleton')
        X_nor = sample_norm
        x_max = max(X_nor[0::2]) + pad_nor
        x_min = min(X_nor[0::2]) - pad_nor
        y_max = max(X_nor[1::2]) + pad_nor
        y_min = min(X_nor[1::2]) - pad_nor
        plt.xlim(x_min,x_max)
        plt.ylim(y_max, y_min)
        plot_skeleton(X_nor, 'ro')
        final_path=os.path.join(path,"{}.png".format(str(name)))
        plt.savefig(final_path)
        plt.close()
    else:
        print("sample is one-dimension array: (28,)")
def plotv2(sample,name,path): # this is for ploting 1 sample
    # e.g the real shape is: (1,2,1,14) ( 2 is x and y) 
    # the expected shape is: (28,)

    if sample.shape[0] == 28:
        sample_norm = sample.reshape(1,28)[0]

        # Plot original coordinates
        pad_ori = 40
        plt.figure(str(sample))
        plt.subplot(121)
        plt.title('Original skeleton')
        X_ori = sample
        x_max = max(X_ori[0::2]) + pad_ori
        x_min = min(i for i in X_ori[0::2] if i > 0) - pad_ori
        y_max = max(X_ori[1::2]) + pad_ori
        y_min = min(j for j in X_ori[1::2] if j > 0) - pad_ori
        plt.xlim(x_min,x_max)
        plt.ylim(y_max, y_min)
        plot_skeleton(X_ori, 'bo')

        # Plot normalized coordinates
        pad_nor = 0.2
        #plt.figure(2)
        plt.subplot(122)
        plt.title('Normalized skeleton')
        X_nor = sample_norm
        x_max = max(X_nor[0::2]) + pad_nor
        x_min = min(X_nor[0::2]) - pad_nor
        y_max = max(X_nor[1::2]) + pad_nor
        y_min = min(X_nor[1::2]) - pad_nor
        plt.xlim(x_min,x_max)
        plt.ylim(y_max, y_min)
        plot_skeleton(X_nor, 'ro')
        final_path=os.path.join(path,"{}.png".format(str(name)))
        plt.savefig(final_path)
        plt.close()

    else:
        print("sample is one-dimension array: (28,)")