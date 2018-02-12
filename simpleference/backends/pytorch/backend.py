import os
import numpy as np

import dill
import torch
from contextlib import contextmanager
from torch.autograd import Variable

from Queue import Queue


class PyTorchPredict(object):
    def __init__(self, model_path, crop=None, gpu_list=[0]):
        assert os.path.exists(model_path), model_path
        if np.isscalar(gpu_list):
            gpu_list = [gpu_list]
        self.model_queue = Queue(maxsize=len(gpu_list))
        for gpu in gpu_list:
            # load and copy the model to all GPUs.
            model = torch.load(model_path, pickle_module=dill)
            model.cuda(gpu)
            self.model_queue.put((gpu, model))
        # validate cropping
        if crop is not None:
            assert isinstance(crop, (list, tuple))
            assert len(crop) == 3
        self.crop = crop

    @contextmanager
    def get_gpu_model(self):
        gpu, model = self.model_queue.get()
        yield gpu, model
        self.model_queue.put((gpu, model))

    def apply_crop(self, out):
        shape_diff = tuple((shape - crop) // 2
                           for shape, crop in zip(out.shape, self.crop))
        bb = tuple(slice(diff, shape - diff) for diff, shape in zip(shape_diff, out.shape))
        return out[bb]

    def __call__(self, input_data):
        assert isinstance(input_data, np.ndarray)
        assert input_data.ndim == 3
        # Note: in the code that follows, the GPU is locked for the 3 steps:
        # CPU -> GPU, GPU inference, GPU -> CPU. It may well be that we get
        # better performance by only locking in step 2, or steps 1-2, or steps
        # 2-3. We should perform this experiment and then choose the best
        # option for our hardware (and then remove this comment! ;)
        with self.get_gpu_model() as gpu, model:
            # 1. Transfer the data to the GPU
            torch_data = Variable(torch.from_numpy(input_data[None, None])
                                  .cuda(gpu), volatile=True)
            print('predicting a block!')
            # 2. Run the model
            predicted_on_gpu = model(torch_data)
            # 3. Transfer the results to the CPU
            out = predicted_on_gpu.cpu().data.numpy().squeeze()
        if self.crop is not None:
            out = self.apply_crop(out)
        return out
