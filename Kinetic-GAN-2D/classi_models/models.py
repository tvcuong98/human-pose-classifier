import torch.nn as nn
class Conv1DPoseClassifier(nn.Module):
    def __init__(self, input_size=28, num_classes=9, drop_out_p=0.1):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=4, stride=2)  # 1 input channel, 128 output channels
        self.dropout1 = nn.Dropout(drop_out_p)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=4, stride=2)  # 128 input channels, 64 output channels
        self.dropout2 = nn.Dropout(drop_out_p)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=4, stride=2)  # 64 input channels, 32 output channels
        self.dropout3 = nn.Dropout(drop_out_p)
        self.fc4 = nn.Linear(32, num_classes)
        #self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.ReLU()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_()
                m.bias.data.zero_()

    def forward(self, x):
        # Add a dimension for the channel (assuming input is 1D)
        x = x.unsqueeze(1)
        
        x = self.sigmoid(self.conv1(x))
        x = self.dropout1(x)
        x = self.sigmoid(self.conv2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.conv3(x))
        x = self.dropout3(x)
        x = x.squeeze(-1)  # Squeeze the last dimension (assumed to be size 1)
        x = self.fc4(x)
        x = self.softmax(x)
        print(x)
        return x
class SimplePoseClassifier(nn.Module):
    def __init__(self, input_size=12, num_classes=9, drop_out_p=0.3):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(drop_out_p)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(drop_out_p)
        self.fc3 = nn.Linear(32, 32)
        self.dropout3 = nn.Dropout(drop_out_p)
        self.fc4 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_()
                m.bias.data.zero_()

    def forward(self, x):
        # print(x.shape)
        x = self.sigmoid(self.fc1(x))
        x = self.dropout1(x)
        x = self.sigmoid(self.fc2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x
class PoseClassifier(nn.Module):
    def __init__(self, input_size=28, num_classes=9, drop_out_p=0.1):
        super().__init__()

        self.block1 = self._create_block(input_size, 128, drop_out_p) # 1024
        self.skip2 = self._create_skip(128, 64) # 1024,512
        self.block2 = self._create_block(128, 64, drop_out_p) # 1024,512
        self.skip3 = self._create_skip(64, 16) # 512,216
        self.block3 = self._create_block(64, 16, drop_out_p) # 512,216
        self.block4 = nn.Sequential(nn.Sigmoid(), nn.Linear(16, num_classes))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_()
                m.bias.data.zero_()

    def _create_block(self, in_features, out_features, drop_out_p):
        block = [
            nn.Linear(in_features, out_features),
            nn.Sigmoid(),
            nn.BatchNorm1d(out_features),
            nn.Dropout(p=drop_out_p),
        ]
        return nn.Sequential(*block)

    def _create_skip(self, in_features, out_features):
        skip = [nn.Linear(in_features, out_features)]
        return nn.Sequential(*skip)

    def forward(self, x):
        # print(x.shape)
        x = self.block1(x)
        x = self.block2(x) + self.skip2(x)
        x = self.block3(x) + self.skip3(x)
        x = self.block4(x)
        return x
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
        # print(x.shape)
        x = self.block1(x)
        x = self.block2(x) + self.skip2(x)
        x = self.block3(x) + self.skip3(x)
        x = self.block4(x)
        return x


