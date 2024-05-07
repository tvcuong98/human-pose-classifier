import torch.nn as nn
from visualization import plot
class HeavyPoseClassifier(nn.Module):
    def __init__(self, input_size=32, num_classes=9, drop_out_p=0.0,hidden_dims=64): #@56
        super().__init__()

        self.block1 = self._create_block(input_size, hidden_dims, drop_out_p) # 1024
        self.skip2 = self._create_skip(hidden_dims, int(hidden_dims/2)) # 1024,512
        self.block2 = self._create_block(hidden_dims, int(hidden_dims/2), drop_out_p) # 1024,512
        self.skip3 = self._create_skip(int(hidden_dims/2), int(hidden_dims/2)) # 512,216
        self.block3 = self._create_block(int(hidden_dims/2), int(hidden_dims/2), drop_out_p) # 512,216
        self.block4 = nn.Sequential(nn.Sigmoid(), nn.Linear(int(hidden_dims/2), num_classes))
        #self.block4 = nn.Sequential(nn.Tanh(),nn.Linear(int(hidden_dims/2), num_classes))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_()
                m.bias.data.zero_()

    def _create_block(self, in_features, out_features, drop_out_p):
        block = [
            nn.Dropout(p=0.0),
            nn.Linear(in_features, out_features),
            #nn.Sigmoid(),
            nn.ReLU(),
            nn.BatchNorm1d(out_features),
            nn.Dropout(p=drop_out_p),
        ]
        return nn.Sequential(*block)

    def _create_skip(self, in_features, out_features):
        skip = [nn.Linear(in_features, out_features)]
        return nn.Sequential(*skip)

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.block1(x)
        x = self.block2(x) + self.skip2(x)
        x = self.block3(x) + self.skip3(x)
        x = self.block4(x)
        return x
# import torch

# data = [[1, 2], [3, 4],[5,6]]
# tensor = torch.tensor(data)
# print(tensor.view(-1)) 