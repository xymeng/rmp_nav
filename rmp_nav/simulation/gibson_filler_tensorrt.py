import os
import numpy as np
from rmp_nav.gibson import assets
from ..common.inference_tensorrt import Inference, get_device_name


assets_file_dir = os.path.dirname(assets.__file__)
name_mapping = {
    'GeForce GTX 1070': 'gtx1070',
    'GeForce GTX 1080': 'gtx1080',
    'GeForce GTX 1080 Ti': 'gtx1080ti',
    'GeForce RTX 2080 Ti': 'rtx2080ti',
    'TITAN X (Pascal)': 'titanxp'
}


class Filler(object):
    def __init__(self, resolution, gpu=0):
        self.gpu = gpu

        device_name = get_device_name(gpu)
        weights_file = 'model_%d_%s.trt' % (resolution, name_mapping[device_name])
        print('filler gpu: %d device: %s filler_model: %s' % (gpu, device_name, weights_file))

        self.engine = Inference(1, gpu)
        assert self.engine.LoadModel(os.path.join(assets_file_dir, weights_file))

        self.mean = np.array([0.57441127, 0.54226291, 0.50356019], np.float32).reshape((3, 1, 1))

    def fill(self, color, depth):
        source_depth = np.ascontiguousarray(np.expand_dims(depth.astype(np.float32) / 128.0, 2).transpose(2, 0, 1))
        source = np.ascontiguousarray(color.astype(np.float32).transpose(2, 0, 1)) / 255.0
        mask = (np.sum(source[:3, :, :], axis=0) > 0).astype(np.float32)[None]  # 1 x res x res
        source += (1.0 - np.broadcast_to(mask, (3,) + mask.shape[1:])) * self.mean
        mask = np.ascontiguousarray(np.concatenate([source_depth, mask], axis=0))
        recon = self.engine.Run([source[None], mask[None]])[0][0]
        return (np.clip(recon, 0.0, 1.0).transpose(1, 2, 0) * 255).astype(np.uint8)


class FillerBatch(object):
    def __init__(self, resolution, max_batch_size=1, gpu=0):
        self.gpu = gpu

        device_name = get_device_name(gpu)
        weights_file = 'model_%d_%s_b%d.trt' % (resolution, name_mapping[device_name], max_batch_size)
        print('filler gpu: %d device: %s filler_model: %s' % (gpu, device_name, weights_file))

        self.engine = Inference(max_batch_size, gpu)
        assert self.engine.LoadModel(os.path.join(assets_file_dir, weights_file))

        self.mean = np.array([0.57441127, 0.54226291, 0.50356019], np.float32).reshape((1, 3, 1, 1))

    def fill(self, colors, depths):
        """
        :param colors: N x H x W x C
        :param depths: N x H x W
        :return:
        """
        source_depths = np.ascontiguousarray(depths[:, :, :, None].astype(np.float32).transpose(0, 3, 1, 2)) / 128.0
        source = np.ascontiguousarray(colors.astype(np.float32).transpose(0, 3, 1, 2)) / 255.0
        mask = (np.sum(source[:, :3, :, :], axis=1) > 0).astype(np.float32)[:, None, :, :]  # N x 1 x res x res
        n, h, w = mask.shape[0], mask.shape[2], mask.shape[3]
        source += (1.0 - np.broadcast_to(mask, (n, 3, h, w))) * self.mean
        mask = np.ascontiguousarray(np.concatenate([source_depths, mask], axis=1))
        recon = self.engine.Run([source, mask])[0]  # N x 3 x H x W
        return (np.clip(recon, 0.0, 1.0).transpose(0, 2, 3, 1) * 255).astype(np.uint8)


if __name__ == '__main__':
    """
    It seems that tensorrt does not benefit much from larger batch size.
    """
    import time
    RES = 128
    GPU = 0
    N_SAMPLES = 8192

    def batch_benchmark():
        batch_size = 8
        filler = FillerBatch(RES, batch_size, GPU)
        dummy_colors = np.random.uniform(0.0, 1.0, (batch_size, RES, RES, 3)).astype(np.float32)
        dummy_depths = np.random.uniform(0.0, 128.0, (batch_size, RES, RES)).astype(np.float32)
        filler.fill(dummy_colors, dummy_depths)

        iter = N_SAMPLES // batch_size
        start_time = time.time()
        for i in range(iter):
            filler.fill(dummy_colors, dummy_depths)
        print((time.time() - start_time) / N_SAMPLES)

    def single_benchmark():
        filler = Filler(RES, GPU)
        dummy_colors = np.random.uniform(0.0, 1.0, (RES, RES, 3)).astype(np.float32)
        dummy_depths = np.random.uniform(0.0, 128.0, (RES, RES)).astype(np.float32)
        filler.fill(dummy_colors, dummy_depths)

        iter = N_SAMPLES
        start_time = time.time()
        for i in range(iter):
            filler.fill(dummy_colors, dummy_depths)
        print((time.time() - start_time) / N_SAMPLES)

    single_benchmark()
    # batch_benchmark()
