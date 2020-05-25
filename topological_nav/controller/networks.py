import torch
from torch import nn
import torch.nn.functional as F


class WaypointRegressor(nn.Module):
    def __init__(self, input_dim=512, mid_dim=None, out_dim=2, init_scale=1.0,
                 bias=True, no_weight_init=False, init_method='ortho', activation='relu'):
        super(WaypointRegressor, self).__init__()

        self.activation = activation

        if mid_dim is None:
            self.fc1 = nn.Linear(input_dim, input_dim // 2, bias=bias)
            self.fc2 = nn.Linear(input_dim // 2, input_dim // 2, bias=bias)
            self.fc3 = nn.Linear(input_dim // 2, out_dim, bias=bias)
        else:
            self.fc1 = nn.Linear(input_dim, mid_dim, bias=bias)
            self.fc2 = nn.Linear(mid_dim, mid_dim, bias=bias)
            self.fc3 = nn.Linear(mid_dim, out_dim, bias=bias)

        if not no_weight_init:
            for layer in (self.fc1, self.fc2, self.fc3):
                if init_method == 'ortho':
                    nn.init.orthogonal_(layer.weight, init_scale)
                if layer.bias is not None:
                    with torch.no_grad():
                        layer.bias.zero_()

    def forward(self, x):
        if self.activation == 'relu':
            ac = F.relu
        elif self.activation == 'tanh':
            ac = torch.tanh
        else:
            raise RuntimeError()

        x = self.fc1(x)
        x = ac(x)
        x = self.fc2(x)
        x = ac(x)
        x = self.fc3(x)
        return x


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


class ProximityRegressor(nn.Module):
    def __init__(self, input_dim=512, output_dim=2, init_scale=1.0, no_weight_init=False):
        super(ProximityRegressor, self).__init__()

        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 2)
        self.fc3 = nn.Linear(input_dim // 2, output_dim)

        if not no_weight_init:
            for layer in (self.fc1, self.fc2, self.fc3):
                nn.init.orthogonal_(layer.weight, init_scale)
                with torch.no_grad():
                    layer.bias.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class HeadingDiffRegressor(nn.Module):
    def __init__(self, input_dim=512, output_dim=2, init_scale=1.0, no_weight_init=False):
        super(HeadingDiffRegressor, self).__init__()

        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 2)
        self.fc3 = nn.Linear(input_dim // 2, output_dim)

        if not no_weight_init:
            for layer in (self.fc1, self.fc2, self.fc3):
                nn.init.orthogonal_(layer.weight, init_scale)
                with torch.no_grad():
                    layer.bias.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


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
        B, D, L = x.size()
        x = self.conv(x)
        x = F.relu(x)
        return x.flatten(1)
