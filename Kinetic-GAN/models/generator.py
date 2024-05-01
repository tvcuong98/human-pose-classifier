import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .init_gan.tgcn import ConvTemporalGraphical
from .init_gan.graph_ntu import graph_ntu
from .init_gan.graph_h36m import Graph_h36m
import numpy as np


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class Mapping_Net(nn.Module):
    def __init__(self, latent=1024, mlp=4):
        super().__init__()

        layers = []
        for i in range(mlp):
            linear = nn.Linear(latent, latent)
            linear.weight.data.normal_()
            linear.bias.data.zero_()
            layers.append(linear)
            layers.append(nn.LeakyReLU(0.2))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Generator(nn.Module):
    
    def __init__(self, in_channels, out_channels, n_classes, t_size, mlp_dim=4, edge_importance_weighting=True, dataset='ntu', **kwargs):
        super().__init__()

        # load graph
        self.graph = graph_ntu() if dataset == 'ntu' else Graph_h36m()
        self.A = [torch.tensor(Al, dtype=torch.float32, requires_grad=False).cuda() for Al in self.graph.As]

        # build networks
        spatial_kernel_size  = [A.size(0) for A in self.A]
        temporal_kernel_size = [3 for i, _ in enumerate(self.A)]
        kernel_size          = (temporal_kernel_size, spatial_kernel_size)
        self.t_size          = t_size

        self.mlp = Mapping_Net(in_channels+n_classes, mlp_dim)
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels+n_classes, 512, kernel_size, 1, graph=self.graph, lvl=3, bn=False, residual=False, up_s=False, up_t=1, **kwargs),
            st_gcn(512, 256, kernel_size, 1, graph=self.graph, lvl=3, up_s=False, up_t=int(t_size/16), **kwargs),
            st_gcn(256, 128, kernel_size, 1, graph=self.graph, lvl=2, bn=False, up_s=True, up_t=int(t_size/16), **kwargs),
            st_gcn(128, 64, kernel_size, 1, graph=self.graph, lvl=2, up_s=False, up_t=int(t_size/8), **kwargs),
            st_gcn(64, 32, kernel_size, 1, graph=self.graph, lvl=1, bn=False, up_s=True, up_t=int(t_size/4), **kwargs),
            st_gcn(32, out_channels, kernel_size, 1, graph=self.graph, lvl=1, up_s=False, up_t=int(t_size/2), **kwargs),
            st_gcn(out_channels, out_channels, kernel_size, 1, graph=self.graph, lvl=0, bn=False, up_s=True, up_t=t_size, tan=True, **kwargs)
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
        

    def forward(self, x, labels, trunc=None): # x.shape = torch.Size([128, 512])
        c = self.label_emb(labels)
        x = torch.cat((c, x), -1) # torch.Size([128, 521])
        w = []
        for i in x:
            w = self.mlp(i).unsqueeze(0) if len(w)==0 else torch.cat((w, self.mlp(i).unsqueeze(0)), dim=0)

        w = self.truncate(w, 1000, trunc) if trunc is not None else w  # Truncation trick on W # torch.Size([128, 521]) (still keeping the same shape)
        x = w.view((*w.shape, 1, 1)) # torch.Size([128, 521, 1, 1])
        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A[gcn.lvl] * importance)
            # After every st_gcn layer:  torch.Size([128, 512, 1, 1]) -> (in_chan,out_chan) = (521,512) , up_s=False, up_t=1
            # After every st_gcn layer:  torch.Size([128, 256, 2, 1]) -> (in_chan,out_chan) = (512,256) , up_s=False, up_t=int(t_size/16)=32/16=2
            # After every st_gcn layer:  torch.Size([128, 128, 2, 2]) -> (in_chan,out_chan) = (256,128) , up_s=True, up_t=int(t_size/16)=32/16=2
            # After every st_gcn layer:  torch.Size([128, 64, 4, 2]) -> (in_chan,out_chan) = (128,64) , up_s=False, up_t=int(t_size/8)=32/8=4
            # After every st_gcn layer:  torch.Size([128, 32, 8, 7]) -> (in_chan,out_chan) = (64,32) , up_s=True, up_t=int(t_size/4)=32/4=8
            # After every st_gcn layer:  torch.Size([128, 2, 16, 7]) -> (in_chan,out_chan) = (32,2) , up_s=False, up_t=int(t_size/2)=32/2=16
            # After every st_gcn layer:  torch.Size([128, 2, 32, 16]) -> (in_chan,out_chan) = (2,2) , up_s=True, up_t=int(t_size)=32

        return x

    def truncate(self, w, mean, truncation):  # Truncation trick on W
        t = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (mean, *w.shape[1:]))))
        w_m = []
        for i in t:
            w_m = self.mlp(i).unsqueeze(0) if len(w_m)==0 else torch.cat((w_m, self.mlp(i).unsqueeze(0)), dim=0)

        m = w_m.mean(0, keepdim=True)

        for i,_ in enumerate(w):
            w[i] = m + truncation*(w[i] - m)

        return w

class st_gcn(nn.Module):

    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                graph=None,
                lvl=3,
                dropout=0,
                bn=True,
                residual=True,
                up_s=False, 
                up_t=64, 
                tan=False):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0][lvl] % 2 == 1
        padding = ((kernel_size[0][lvl] - 1) // 2, 0)
        self.graph, self.lvl, self.up_s, self.up_t, self.tan = graph, lvl, up_s, up_t, tan
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, # perform convolution on the C dim, decrease its size , originally at 512, final at 2 (x,y)
                                        kernel_size[1][lvl])
        tcn = [nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0][lvl], 1), #kernel_size is ([3, 3, 3, 3], [3, 3, 3, 3])
                (stride, 1), # stride is always 1
                padding,
            )] # dont change the shape of the tensor
        
        tcn.append(nn.BatchNorm2d(out_channels)) if bn else None

        self.tcn = nn.Sequential(*tcn)


        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.noise = NoiseInjection(out_channels)

        self.l_relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh   = nn.Tanh()

    def forward(self, x, A):
        # print("Original shape: ", x.shape)
        x = self.upsample_s(x) if self.up_s else x # upsample the V dimension-spatio dim, 1->2->7->16, at every st_gcn layer in which up_s = True
        # print("After upsample_s, before interpolate : ", x.shape)
        x = F.interpolate(x, size=(self.up_t,x.size(-1)))  # Exactly like nn.Upsample, here it increase the T in (N,C,T,V), according to up_t
        # print("After interpolate, before ConvTemporalGraphical : ", x.shape)
        res = self.residual(x)
        x, A = self.gcn(x, A) # perform convolution on the C dim, decrease its size , originally at 512, final at 2 (x,y)
        # print("After ConvTemporalGraphical, before tcn  : ", x.shape)
        x    = self.tcn(x) + res # dont change the shape of the tensor
        # print("After tcn, before  Noise Inject  : ", x.shape)
        # Noise Inject
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device='cuda:0')
        x     = self.noise(x, noise) # dont change the shape of the tensor
        # print("After Noise Inject  : ", x.shape)
        return self.tanh(x) if self.tan else self.l_relu(x), A



    def upsample_s(self, tensor): # upsample the V dimension-spatio dim, 1->2->7->16, at every st_gcn layer in which up_s = True

        ids  = []
        mean = []
        for umap in self.graph.mapping[self.lvl]:
            ids.append(umap[0])
            tmp = None
            for nmap in umap[1:]:
                tmp = torch.unsqueeze(tensor[:, :, :, nmap], -1) if tmp == None else torch.cat([tmp, torch.unsqueeze(tensor[:, :, :, nmap], -1)], -1)

            mean.append(torch.unsqueeze(torch.mean(tmp, -1) / (2 if self.lvl==2 else 1), -1))

        for i, idx in enumerate(ids): tensor = torch.cat([tensor[:,:,:,:idx], mean[i], tensor[:,:,:,idx:]], -1)


        return tensor
    # Overall Functionality
    # The upsample_s function appears to perform a specific upsampling operation designed to:
    # Group nodes: Based on the graph's structure and connectivity.
    # Aggregate features: Calculate averages or summaries of features within these node groups.
    # Insert aggregated features: Expand the spatial dimension of the feature representation by strategically inserting these aggregated features