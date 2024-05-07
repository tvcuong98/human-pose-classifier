import torch
import numpy as np
import torch.nn as nn
def has_nan(input_array):
    """Checks for NaN values in a NumPy array or a PyTorch tensor.

    Args:
        input_array: The NumPy array or PyTorch tensor to check.

    Returns:
        True if the input contains at least one NaN value, False otherwise.
    """

    if isinstance(input_array, np.ndarray):
        return np.isnan(input_array).any()
    elif isinstance(input_array, torch.Tensor):
        # Safely handle tensors on CPU or GPU
        return torch.isnan(input_array.to('cpu')).any().item()
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
def has_zero(input_array):
    """Checks for zero values in a NumPy array or a PyTorch tensor.

    Args:
        input_array: The NumPy array or PyTorch tensor to check.

    Returns:
        True if the input contains at least one zero value, False otherwise.
    """

    if isinstance(input_array, np.ndarray):
        return (input_array == 0).any()  
    elif isinstance(input_array, torch.Tensor):
        # Safely handle tensors on CPU or GPU
        return (input_array.to('cpu') == 0).any().item() 
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias) 
def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad
def get_BoneVecbypose2d(x, num_joints=16):
    '''
    :explain: convert 2D point to bone vector
    :param x: N x number of joint x 2
    :return: N x number of bone x 2  number of bone = number of joint - 1
    '''
    Ct = torch.Tensor([
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 1
        [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 2
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 3
        [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 4
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4 5
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5 6
        [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 7
        [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],  # 7 8
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],  # 8 9
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0],  # 8 10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],  # 10 11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],  # 11 12
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0],  # 8 13
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],  # 13 14
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],  # 14 15
    ]).transpose(1, 0)

    Ct = Ct.to(x.device)
    C = Ct.repeat([x.size(0), 1, 1]).view(-1, num_joints, num_joints - 1)
    pose3 = x.permute(0, 2, 1).contiguous()  # 这里16x3变成3x16的话 应该用permute吧
    B = torch.matmul(pose3, C)
    B = B.permute(0, 2, 1)  # back to N x 15 x 3
    return B
def get_pose2dbyBoneVec(bones, num_joints=16):
    '''
    :explain: convert bone vect to pose2d， get_BoneVecbypose2d
    :param bones: N x number of bone x 2 , knowing number of bone = number of joint - 1
    :return: N x number of joint x 2
    '''
    Ctinverse = torch.Tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 basement
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 1
        [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 2
        [-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 3
        [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 4
        [0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4 5
        [0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5 6
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 7
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0],  # 7 8
        [0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0],  # 8 9
        [0, 0, 0, 0, 0, 0, -1, -1, 0, -1, 0, 0, 0, 0, 0],  # 8 10
        [0, 0, 0, 0, 0, 0, -1, -1, 0, -1, -1, 0, 0, 0, 0],  # 10 11
        [0, 0, 0, 0, 0, 0, -1, -1, 0, -1, -1, -1, 0, 0, 0],  # 11 12
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, 0, 0],  # 8 13
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0],  # 13 14
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, -1],  # 14 15
    ]).transpose(1, 0)

    Ctinverse = Ctinverse.to(bones.device)
    C = Ctinverse.repeat([bones.size(0), 1, 1]).view(-1, num_joints - 1, num_joints)
    bonesT = bones.permute(0, 2, 1).contiguous()
    pose2d = torch.matmul(bonesT, C)
    pose2d = pose2d.permute(0, 2, 1).contiguous()  # back to N x 16 x 3
    return pose2d
def get_bone_lengthbypose2d(x, bone_dim=2):
    '''
    :param x: N x number of joint x 2
    :param bone_dim: dim=2, 
    :since N x number of bone x 2, the bone_dim is interpret as the "coordinates" of the bone, which is the xy value at the last dim 
    
    :return: N x number of bone x 1 , number of bone = number of joint - 1
    :since only need 1 element to express the "length", the depth of last dim is 1
    '''
    bonevec = get_BoneVecbypose2d(x)
    bones_length = torch.norm(bonevec, dim=2, keepdim=True)
    bones_length = torch.where(bones_length == 0, bones_length + 0.0001,bones_length)
    return bones_length
def get_bone_unit_vecbypose2d(x, num_joints=16, bone_dim=2):
    '''
    :param x: N x number of joint x 2
    :param bone_dim: bone_dim=2, 
    :param bone_dim: num_joints=16, 

    :return: N x number of bone x 2
    :explain : number of bone = number of joint - 1
    '''
    bonevec = get_BoneVecbypose2d(x) # N,15,2
    bonelength = get_bone_lengthbypose2d(x)  # N,15,1
    bonelength = torch.where(bonelength == 0, bonelength + 0.0001,bonelength) # avoid dividing by 0
    bone_unitvec = bonevec / bonelength # this is where it gets nan -> dividing by 0      
    return bone_unitvec
def blaugment9to15(x, bl, blr, num_bone=15):
    '''
    this function convert 9 blr to 15 blr, and apply to bl
    x : b x joints x 2
    bl: b x joints-1 x 1
    blr: b x 9 x 1
    out: pose3d b x joints x 2
    '''
    blr9to15 = torch.Tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
        [0, 1, 0, 0, 0, 0, 0, 0, 0],  # 2
        [0, 0, 1, 0, 0, 0, 0, 0, 0],  # 3
        [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
        [0, 1, 0, 0, 0, 0, 0, 0, 0],  # 5
        [0, 0, 1, 0, 0, 0, 0, 0, 0],  # 6
        [0, 0, 0, 1, 0, 0, 0, 0, 0],  # 7
        [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 8
        [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 9
        [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 10
        [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 11
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 12
        [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 13
        [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 14
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 15
    ]).transpose(1, 0)  # 9 x 15 matrix

    blr9to15 = blr9to15.to(blr.device)
    blr9to15 = blr9to15.repeat([blr.size(0), 1, 1]).view(blr.size(0), 9, 15)
    blr_T = blr.permute(0, 2, 1).contiguous()
    blr_15_T = torch.matmul(blr_T, blr9to15)
    blr_15 = blr_15_T.permute(0, 2, 1).contiguous()  # back to N x 15 x 1

    # convert 2d pose to root relative
    root = x[:, :1, :] * 1.0
    x = x - x[:, :1, :]

    # extract length, unit bone vec
    bones_unit = get_bone_unit_vecbypose2d(x)

    # prepare a bone length list for augmentation.
    bones_length = torch.mul(bl, blr_15) + bl  # res
    modifyed_bone = bones_unit * bones_length

    # convert bone vec back to pose3d
    out = get_pose2dbyBoneVec(modifyed_bone)

    return out + root  # return the pose with position information.
