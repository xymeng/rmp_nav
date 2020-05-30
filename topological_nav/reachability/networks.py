import torch
from torch import nn
import torch.nn.functional as F


class ReachabilityRegressor(nn.Module):
    def __init__(self, input_dim=512, init_scale=1.0, bias=True, no_weight_init=False):
        super(ReachabilityRegressor, self).__init__()

        self.fc1 = nn.Linear(input_dim, input_dim // 2, bias=bias)
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 2, bias=bias)
        self.fc3 = nn.Linear(input_dim // 2, 1, bias=bias)

        if not no_weight_init:
            for layer in (self.fc1, self.fc2, self.fc3):
                nn.init.orthogonal_(layer.weight, init_scale)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    with torch.no_grad():
                        layer.bias.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # Output are logits. Need to apply sigmoid to get values in (0, 1)
        return x


class ImagePairEncoderV2(nn.Module):
    def __init__(self, init_scale=1.0, bias=True, no_weight_init=False):
        super(ImagePairEncoderV2, self).__init__()

        # Input: 9 x 64 x 64
        # img1, img2, img1 - img2 total 9 channels
        self.conv1 = nn.Conv2d(9, 64, kernel_size=5, stride=2, bias=bias)
        # 64 x 30 x 30
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, bias=bias)
        # 128 x 13 x 13
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, bias=bias)
        # 256 x 5 x 5
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=1, bias=bias)
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
