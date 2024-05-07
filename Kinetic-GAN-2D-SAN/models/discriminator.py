import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .init_gan.tgcn import ConvTemporalGraphical
from .init_gan.graph_ntu import graph_ntu
from .init_gan.graph_h36m import Graph_h36m
import numpy as np

class Mapping_Net(nn.Module):
    def __init__(self, latent=1024, mlp=4):
        super().__init__()

        layers = []
        for i in range(mlp):
            linear = nn.Linear(latent, latent)
            linear.weight.data.normal_()
            linear.bias.data.zero_()
            layers.append(linear)
            # layers.append(nn.LeakyReLU(0.2))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
class Discriminator(nn.Module):
    
    def __init__(self, in_channels, n_classes, t_size, latent, mlp_dim=1, edge_importance_weighting=True, dataset='ntu', **kwargs):
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
            st_gcn(in_channels+n_classes, 32, kernel_size, 1, graph=self.graph, lvl=0, dw_s=True, dw_t=1, residual=False, **kwargs),
            st_gcn(32, 64, kernel_size, 1, graph=self.graph, lvl=1, dw_s=False, dw_t=1, **kwargs),
            st_gcn(64, 128, kernel_size, 1, graph=self.graph, lvl=1, dw_s=True, dw_t=1, **kwargs),
            st_gcn(128, 256, kernel_size, 1, graph=self.graph, lvl=2, dw_s=False, dw_t=1, **kwargs),
            st_gcn(256, 512, kernel_size, 1, graph=self.graph, lvl=2, dw_s=True, dw_t=1,  **kwargs),
            st_gcn(512, latent, kernel_size, 1, graph=self.graph, lvl=3, dw_s=False, dw_t=1,  **kwargs),
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
        # some mlp layer
        # self.mlp = Mapping_Net(latent, mlp_dim)
        # fcn for prediction
        self.fcn_w = nn.Parameter(torch.randn(1, latent)) # nn.Linear(1,512) 

    def forward(self, x, labels,flg_train: bool):
        
        N, C, T, V = x.size()


        c = self.label_emb(labels)
        c = c.view(c.size(0), c.size(1), 1, 1).repeat(1, 1, T, V)

        x = torch.cat((c, x), 1)
        
        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A[gcn.lvl] * importance)

        
        # # global pooling
        # x = F.avg_pool2d(x, x.size()[2:])
        # print(x.shape)
        # Conv instead of global pooling
        # x = self.conv(x)
        x = x.view(N, -1) #(N,512)
        # mlp layers
        # x = self.mlp(x)
        h_feature = x #(N,512)
        weights = self.fcn_w
        direction = F.normalize(weights,dim=1) # Normalize the last layer
        scale = torch.norm(weights,dim=1).unsqueeze(1)
        h_feature = h_feature*1 # For keep the scale
        # prediction
        if flg_train: # for discriminator training
            validity_fun = (h_feature.detach() * direction).sum(dim=1)
            validity_dir = (h_feature * direction.detach()).sum(dim=1)
            validity = dict(fun=validity_fun,dir=validity_dir)
        else: # for generator training or inference
            validity = (h_feature * direction).sum(dim=1)

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
                dw_s=False, dw_t=1):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0][lvl] % 2 == 1
        padding = ((kernel_size[0][lvl] - 1) // 2, 0)
        self.graph, self.lvl, self.dw_s, self.dw_t = graph, lvl, dw_s, dw_t
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                        kernel_size[1][lvl])

        self.tcn = nn.utils.spectral_norm(
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0][lvl], 1),
                (stride, 1),
                padding,
            )
        )


        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.utils.spectral_norm(
            nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=(stride, 1)
                        )
            )


        self.l_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, A):
        

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x    = self.tcn(x) + res

        x = self.downsample_s(x) if self.dw_s else x
        
        # x = F.interpolate(x, size=(self.dw_t,x.size(-1)))  # Exactly like nn.Upsample

        return self.l_relu(x), A


    def downsample_s(self, tensor):
        keep = self.graph.map[self.lvl+1][:,1]

        return tensor[:,:,:,keep]