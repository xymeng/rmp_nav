from __future__ import print_function
from past.builtins import xrange

import torch
from torch import nn
import torch.nn.functional as F


class ImageEncoderV3(nn.Module):
    def __init__(self, output_dim=512, init_scale=1.0, residual_link=False):
        super(ImageEncoderV3, self).__init__()
        self.residual_link = residual_link

        # Input: 3 x 64 x 64
        self.conv1 = nn.Conv2d(3, output_dim // 8, kernel_size=5, stride=2)
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

        for layer in (self.conv1, self.conv2, self.conv3, self.conv4):
            nn.init.orthogonal_(layer.weight, init_scale)

    def forward(self, imgs):
        if self.residual_link:
            x = F.relu(self.conv1(imgs))
            x = F.relu(self.res_fc1(x[:, :, 2:-2, 2:-2]) + self.conv2(x))
            x = F.relu(self.res_fc2(x[:, :, 2:-2, 2:-2]) + self.conv3(x))
            x = F.relu(self.res_fc3(x[:, :, 2:-2, 2:-2]) + self.conv4(x))
        else:
            x = F.relu(self.conv1(imgs))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))

        return x.view(x.size(0), -1)


class WaypointEncoder(nn.Module):
    def __init__(self, init_scale=1.0):
        super(WaypointEncoder, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        for layer in (self.fc1, self.fc2):
            nn.init.orthogonal_(layer.weight, init_scale)
            with torch.no_grad():
                layer.bias.zero_()

    def forward(self, wp):
        x = F.relu(self.fc1(wp))
        x = F.relu(self.fc2(x))
        return x


class VelocityEncoder(nn.Module):
    def __init__(self, init_scale=1.0):
        super(VelocityEncoder, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        for layer in (self.fc1, self.fc2):
            nn.init.orthogonal_(layer.weight, init_scale)
            with torch.no_grad():
                layer.bias.zero_()

    def forward(self, vel):
        x = F.relu(self.fc1(vel))
        x = F.relu(self.fc2(x))
        return x


class AngularVelocityEncoder(nn.Module):
    def __init__(self, init_scale=1.0):
        super(AngularVelocityEncoder, self).__init__()
        self.fc1 = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256, 256)
        for layer in (self.fc1, self.fc2):
            nn.init.orthogonal_(layer.weight, init_scale)
            with torch.no_grad():
                layer.bias.zero_()

    def forward(self, angular_vel):
        x = F.relu(self.fc1(angular_vel))
        x = F.relu(self.fc2(x))
        return x


class RMPRegressor(nn.Module):
    def __init__(self, input_dim=1280, n_control_points=12, init_scale=1.0):
        super(RMPRegressor, self).__init__()
        self.n_control_points = n_control_points

        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, init_scale)
                with torch.no_grad():
                    m.bias.zero_()

        self.accel_regressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, n_control_points * 2))

        self.metric_regressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, n_control_points * 3))

        self.apply(weight_init)

    def forward(self, state_feature):
        accel = self.accel_regressor(state_feature)
        metric = self.metric_regressor(state_feature)

        metric_full = metric.new_zeros((metric.size(0), self.n_control_points * 4))
        for j in xrange(self.n_control_points):
            metric_full[:, j * 4] = metric[:, j * 3]
            metric_full[:, j * 4 + 1] = metric[:, j * 3 + 1]
            metric_full[:, j * 4 + 2] = metric[:, j * 3 + 1]
            metric_full[:, j * 4 + 3] = metric[:, j * 3 + 2]

        return accel, metric_full


class ResNetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels, stride, downsample=None):
        super(ResNetBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)

        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(out), inplace=True)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out, inplace=True)
        return out


class ResidualBlocks(nn.Module):
    def __init__(self, in_channels, layer_strides, layer_out_channels, layer_bottleneck_channels):
        super(ResidualBlocks, self).__init__()

        n_layer = len(layer_strides)
        assert n_layer == len(layer_out_channels)
        assert n_layer == len(layer_bottleneck_channels)

        blocks = []
        for i in range(n_layer):
            stride = layer_strides[i]

            if i == 0:
                in_ch = in_channels
            else:
                in_ch = layer_out_channels[i - 1]

            out_ch = layer_out_channels[i]

            if stride != 1:
                downsample = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
            else:
                downsample = None

            blocks.append(ResNetBottleneck(
                in_ch, out_ch, layer_bottleneck_channels[i], layer_strides[i], downsample))

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Residual6BlocksFeature(nn.Module):
    def __init__(self, input_channels=3, init_method='default'):
        super(Residual6BlocksFeature, self).__init__()
        self.resblocks = ResidualBlocks(
            input_channels, [2, 1, 2, 1, 2, 1], [64, 64, 64, 64, 64, 64], [32, 32, 32, 32, 32, 32])
        self.fc = nn.Linear(4096, 512)

        def orthogonal_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.zero_()

        if init_method == 'orthogonal':
            with torch.no_grad():
                self.apply(orthogonal_init)
        else:
            assert init_method == 'default'

    def forward(self, input):
        out = self.resblocks(input).view(input.size(0), -1)
        out = self.fc(out)
        out = torch.tanh(out)
        return out
