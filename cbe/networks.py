import torch
from torch import nn
import torch.nn.functional as F


class ImagePairEncoderV2(nn.Module):
    def __init__(self, init_scale=1.0, no_weight_init=False):
        super(ImagePairEncoderV2, self).__init__()

        # Input: 9 x 64 x 64
        # img1, img2, img1 - img2 total 9 channels
        self.conv1 = nn.Conv2d(9, 64, kernel_size=5, stride=2)
        # 64 x 30 x 30
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        # 128 x 13 x 13
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        # 256 x 5 x 5
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=1)
        # 512 x 1 x 1

        if not no_weight_init:
            for layer in (self.conv1, self.conv2, self.conv3, self.conv4):
                nn.init.orthogonal_(layer.weight, init_scale)

    def forward(self, src_imgs, dst_imgs):
        imgs = torch.cat([src_imgs, dst_imgs, src_imgs - dst_imgs], dim=1)
        x = F.relu(self.conv1(imgs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x.view(x.size(0), -1)


class ImageEncoderV3(nn.Module):
    def __init__(self, input_channels=3, output_dim=512, init_scale=1.0, residual_link=False,
                 no_weight_init=False, init_method='ortho', activation='relu'):
        super(ImageEncoderV3, self).__init__()
        self.residual_link = residual_link
        self.activation = activation

        # Input: 3 x 64 x 64
        self.conv1 = nn.Conv2d(input_channels, output_dim // 8, kernel_size=5, stride=2)
        if residual_link:
            self.res_fc1 = nn.Conv2d(output_dim // 8, output_dim // 4, kernel_size=1, stride=2)

        # 30 x 30
        self.conv2 = nn.Conv2d(output_dim // 8, output_dim // 4, kernel_size=5, stride=2)
        if residual_link:
            self.res_fc2 = nn.Conv2d(output_dim // 4, output_dim // 2, kernel_size=1, stride=2)

        # 13 x 13
        self.conv3 = nn.Conv2d(output_dim // 4, output_dim // 2, kernel_size=5, stride=2)
        if residual_link:
            self.res_fc3 = nn.Conv2d(output_dim // 2, output_dim, kernel_size=1, stride=1)

        # 5 x 5
        self.conv4 = nn.Conv2d(output_dim // 2, output_dim, kernel_size=5, stride=1)
        # 1 x 1

        if not no_weight_init:
            for layer in (self.conv1, self.conv2, self.conv3, self.conv4):
                if init_method == 'ortho':
                    nn.init.orthogonal_(layer.weight, init_scale)
                elif init_method == 'normal':
                    nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                elif init_method == 'xavier_normal':
                    nn.init.xavier_normal_(layer.weight, 1.0)
                else:
                    assert init_method == 'default'

    def forward(self, imgs):
        if self.activation == 'relu':
            ac = F.relu
        elif self.activation == 'tanh':
            ac = torch.tanh
        else:
            raise RuntimeError()

        if self.residual_link:
            x = ac(self.conv1(imgs))
            x = ac(self.res_fc1(x[:, :, 2:-2, 2:-2]) + self.conv2(x))
            x = ac(self.res_fc2(x[:, :, 2:-2, 2:-2]) + self.conv3(x))
            x = ac(self.res_fc3(x[:, :, 2:-2, 2:-2]) + self.conv4(x))
        else:
            x = ac(self.conv1(imgs))
            x = ac(self.conv2(x))
            x = ac(self.conv3(x))
            x = ac(self.conv4(x))

        return x.view(x.size(0), -1)


class ImageEncoderV4(nn.Module):
    """
    Outputs a 5 x 5 x 32 feature map that preserves spatial information.
    """
    def __init__(self, input_channels=3, init_scale=1.0,
                 no_weight_init=False, init_method='ortho', activation='relu'):
        super(ImageEncoderV4, self).__init__()
        self.activation = activation

        # Input: 3 x 64 x 64
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=2)
        # 30 x 30
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # 13 x 13
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # 5 x 5

        if not no_weight_init:
            for layer in (self.conv1, self.conv2, self.conv3):
                if init_method == 'ortho':
                    nn.init.orthogonal_(layer.weight, init_scale)
                elif init_method == 'normal':
                    nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                elif init_method == 'xavier_normal':
                    nn.init.xavier_normal_(layer.weight, 1.0)
                else:
                    assert init_method == 'default'

    def forward(self, imgs):
        if self.activation == 'relu':
            ac = F.relu
        elif self.activation == 'tanh':
            ac = torch.tanh
        else:
            raise RuntimeError()
        x = ac(self.conv1(imgs))
        x = ac(self.conv2(x))
        x = ac(self.conv3(x))  # output_dim x 5 x 5

        return x


class FeatureMapPairEncoderV2(nn.Module):
    def __init__(self, init_scale=1.0, no_weight_init=False):
        super(FeatureMapPairEncoderV2, self).__init__()

        # Input: 96 x 5 x 5
        self.conv1 = nn.Conv2d(96, 256, kernel_size=3, stride=1)
        # 3 x 3
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1)

        if not no_weight_init:
            for layer in (self.conv1, self.conv2):
                nn.init.orthogonal_(layer.weight, init_scale)

    def forward(self, src_imgs, dst_imgs):
        imgs = torch.cat([src_imgs, dst_imgs, src_imgs - dst_imgs], dim=1)
        x = F.relu(self.conv1(imgs))
        x = F.relu(self.conv2(x))
        return x.view(x.size(0), -1)


class ConvEncoder(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, kernel_size=1, init_scale=1.0,
                 no_weight_init=False):
        super(ConvEncoder, self).__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size)
        if not no_weight_init:
            for layer in (self.conv,):
                nn.init.orthogonal_(layer.weight, init_scale)
                with torch.no_grad():
                    layer.bias.zero_()

    def forward(self, x):
        # Input size: batch_size x feature_dim x seq_len
        x = self.conv(x)
        x = F.relu(x)
        return x.flatten(1)


class Recurrent(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, init_scale=1.0, no_weight_init=False):
        super(Recurrent, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = False
        self.recurrent = nn.LSTM(input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers)

        if not no_weight_init:
            weight_inited = False
            for name, param in self.recurrent.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, init_scale)
                    weight_inited = True
                elif 'bias' in name:
                    with torch.no_grad():
                        param.zero_()
            assert weight_inited

    def forward(self, input, hidden_state=None):
        return self.recurrent(input, hidden_state)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim=3, mid_dim=64, output_dim=64, init_scale=1.0, no_weight_init=False):
        super(MLPEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)

        if not no_weight_init:
            for layer in (self.fc1, self.fc2):
                nn.init.orthogonal_(layer.weight, init_scale)
                with torch.no_grad():
                    layer.bias.zero_()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class MLPRegressorV2(nn.Module):
    def __init__(self, dims, init_scale=1.0, no_weight_init=False):
        super(MLPRegressorV2, self).__init__()

        self.fcs = []
        for i in range(len(dims) - 1):
            self.fcs.append(nn.Linear(dims[i], dims[i + 1]))
        self.fcs = nn.ModuleList(self.fcs)

        if not no_weight_init:
            for layer in self.fcs:
                nn.init.orthogonal_(layer.weight, init_scale)
                with torch.no_grad():
                    layer.bias.zero_()

    def forward(self, x):
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        x = self.fcs[-1](x)
        return x


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, init_scale=1.0, no_weight_init=False):
        super(GRUCell, self).__init__()

        self.recurrent = nn.GRUCell(input_size, hidden_size)

        if not no_weight_init:
            for name, param in self.recurrent.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, init_scale)
                    weight_inited = True
                elif 'bias' in name:
                    with torch.no_grad():
                        param.zero_()
            assert weight_inited

    def forward(self, x, h=None):
        return self.recurrent(x, h)
