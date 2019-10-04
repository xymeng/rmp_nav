from __future__ import print_function
import torch
import torch.nn as nn
import os
import numpy as np

from rmp_nav.gibson.learn.completion import CompletionNet
from rmp_nav.gibson import assets

assets_file_dir = os.path.dirname(assets.__file__)


class Filler(object):
    def __init__(self, resolution, gpu=0):
        self.gpu = gpu

        print('filler gpu:', gpu)

        with torch.cuda.device(self.gpu):
            comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
            comp = torch.nn.DataParallel(comp, device_ids=[gpu])
            comp.load_state_dict(
                torch.load(os.path.join(assets_file_dir, "model_{}.pth".format(resolution)),
                           map_location=lambda storage, location: storage))
            self.model = comp.module
            self.model.train(False)

            self.mean = torch.as_tensor(
                np.array([0.57441127, 0.54226291, 0.50356019]), dtype=torch.float, device='cuda')
            self.mean = self.mean.view(3, 1, 1).clone().repeat(1, resolution, resolution)

    def fill(self, color, depth):
        with torch.cuda.device(self.gpu):
            with torch.no_grad():
                source_depth = torch.as_tensor(
                    np.ascontiguousarray(np.expand_dims(depth, 2).transpose(2, 0, 1)),
                    dtype=torch.float,
                    device='cuda') / 128.0

                # source_depth = source_depth.cuda(async=True).float() / 128.0

                source = torch.as_tensor(
                    np.ascontiguousarray(color.transpose(2, 0, 1)),
                    dtype=torch.float,
                    device='cuda') / 255.0

                mask = (torch.sum(source[:3, :, :], 0) > 0).float().unsqueeze(0)
                source += (1 - mask.repeat(3, 1, 1)) * self.mean

                mask = torch.cat([source_depth, mask], 0)
                recon = self.model(source.unsqueeze(0), mask.unsqueeze(0))

                show2 = recon.data.clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0)
                return (show2[:] * 255).astype(np.uint8)
