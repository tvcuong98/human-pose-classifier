import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .init_gan.tgcn import ConvTemporalGraphical
from .init_gan.graph_ntu import graph_ntu
from .init_gan.graph_h36m import Graph_h36m
import numpy as np


class Discriminator(nn.Module):
    
    def __init__(self, in_channels, n_classes, t_size, latent, edge_importance_weighting=True, dataset='ntu', **kwargs):
        super().__init__()

        # load graph
        self.graph = graph_ntu() if dataset == 'ntu' else Graph_h36m()
        self.A = [torch.tensor(Al, dtype=torch.float32, requires_grad=False).cuda() for Al in self.graph.As]

        # build networks
        spatial_kernel_size  = [A.size(0) for A in self.A]
        temporal_kernel_size = [3 for _ in self.A]
        kernel_size          = (temporal_kernel_size, spatial_kernel_size)
        self.t_size          = t_size

        #kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels+n_classes, 32, kernel_size, 1, graph=self.graph, lvl=0, dw_s=True, dw_t=t_size, residual=False, **kwargs),
            st_gcn(32, 64, kernel_size, 1, graph=self.graph, lvl=1, dw_s=False, dw_t=t_size, **kwargs),
            st_gcn(64, 128, kernel_size, 1, graph=self.graph, lvl=1, dw_s=True, dw_t=int(t_size/2), **kwargs),
            st_gcn(128, 256, kernel_size, 1, graph=self.graph, lvl=2, dw_s=False, dw_t=int(t_size/4), **kwargs),
            st_gcn(256, 512, kernel_size, 1, graph=self.graph, lvl=2, dw_s=True, dw_t=int(t_size/8),  **kwargs),
            st_gcn(512, latent, kernel_size, 1, graph=self.graph, lvl=3, dw_s=False, dw_t=int(t_size/16),  **kwargs),
        ))


        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A[i.lvl].size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.label_emb = nn.Embedding(n_classes, n_classes)
        # adding conv layer
        self.conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(2, 1), stride=(2, 1), padding=0)
        # fcn for prediction
        self.fcn = nn.Linear(latent, 1)

    def forward(self, x, labels):
        
        N, C, T, V = x.size() # torch.Size([128, 2, 32, 16])


        c = self.label_emb(labels)
        c = c.view(c.size(0), c.size(1), 1, 1).repeat(1, 1, T, V)
        x = torch.cat((c, x), 1) # torch.Size([128, 11, 32, 16]) # 9 class embedding have been concatenated into the xy channel
        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A[gcn.lvl] * importance)
            # There are 6 layers of st_gcn_networks, and the output after it pass through each of them:
            # torch.Size([128, 32, 32, 7]) -> dw_s=True, dw_t=t_size , (channel_in,channel_out) = (11, 32)
            # torch.Size([128, 64, 32, 7] -> dw_s=False, dw_t=t_size , (channel_in,channel_out) = (32, 64)
            # torch.Size([128, 128, 16, 2]) -> dw_s=True, dw_t=int(t_size/2) , (channel_in,channel_out) = (64, 128)
            # torch.Size([128, 256, 8, 2]) -> dw_s=False, dw_t=int(t_size/4) , (channel_in,channel_out) = (128, 256)
            # torch.Size([128, 512, 4, 1]) -> dw_s=True, dw_t=int(t_size/8) , (channel_in,channel_out) = (256, 512)
            # torch.Size([128, 512, 2, 1]) -> dw_s=False, dw_t=int(t_size/16) , (channel_in,channel_out) = (512, 512)

        
        # global pooling
        # x = F.avg_pool2d(x, x.size()[2:]) # torch.Size([128, 512, 2, 1]) -> torch.Size([128, 512, 1])
        # conv layer instead of pooling
        x = self.conv(x)
        x = x.view(N, -1) # torch.Size([128, 512, 1]) -> torch.Size([128, 512])

        # prediction
        validity = self.fcn(x)

        return validity

    

class st_gcn(nn.Module):

    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                graph=None,
                lvl=3,
                dropout=0,
                residual=True,
                dw_s=False, dw_t=64):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0][lvl] % 2 == 1
        padding = ((kernel_size[0][lvl] - 1) // 2, 0)
        self.graph, self.lvl, self.dw_s, self.dw_t = graph, lvl, dw_s, dw_t
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                        kernel_size[1][lvl])  # the ConvTemporalGraphical will increase the channel , increase C in (N,C,T,V)

        self.tcn = nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0][lvl], 1),
                (stride, 1),
                padding,
            ) # the tcn dont change the shape of the tensor


        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=(stride, 1)
                    )


        self.l_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, A):
        
        # print("Before ConvTemporalGraphical shape: ", x.shape)
        res = self.residual(x)
        x, A = self.gcn(x, A)
        # print("After ConvTemporalGraphical, before tcn shape: ", x.shape)
        x    = self.tcn(x) + res
        # print("After tcn, before downsample_s shape: ", x.shape)
        x = self.downsample_s(x) if self.dw_s else x # downsample_s decrease the V dimension in (N,C,T,V), if dw_s is True in the st_gcn params
                                                     # 16 -> 7 -> 2 -> 1
        # print("After downsample_s, before F.interpolate shape: ", x.shape)
        x = F.interpolate(x, size=(self.dw_t,x.size(-1)))  # F.interpolate decrease the T dimension in (N,C,T,V), according to the value of dw_t
        # print("After F.interpolate shape : ", x.shape)
        return self.l_relu(x), A


    def downsample_s(self, tensor): # downsample_s decrease the V dimension in (N,C,T,V), if dw_s is True in the st_gcn params
                                    # 16 -> 7 -> 2 -> 1
        keep = self.graph.map[self.lvl+1][:,1]

        return tensor[:,:,:,keep]