import os
import sys
import time
import json

from simpleference.inference.inference import run_inference_n5
from simpleference.backends.pytorch import PyTorchPredict
from simpleference.backends.pytorch.preprocess import preprocess


def single_gpu_inference(sample, gpu):
    raw_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
    out_file = '/groups/saalfeld/home/papec/Work/neurodata_hdd/inference_tests/torch_master_test_sample%s.n5' % sample
    assert os.path.exists(out_file)

    model_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/networks/neurofire'
    model_path = os.path.join(model_path, 'criteria_exps/sorensen_dice_unweighted/Weights/networks/model.pytorch')

    offset_file = './offsets_sample%s/list_gpu_%i.json' % (sample, 0)
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    input_shape = (84, 270, 270)
    output_shape = (56, 56, 56)
    prediction = PyTorchPredict(model_path,
                                crop=output_shape,
                                gpu=gpu)

    t_predict = time.time()
    run_inference_n5(prediction,
                     preprocess,
                     raw_path,
                     out_file,
                     offset_list,
                     input_key='raw',
                     target_keys=('affs_xy', 'affs_z'),
                     input_shape=input_shape,
                     output_shape=output_shape,
                     only_nn_affs=True,
                     num_cpus=4 * len(gpu))
    t_predict = time.time() - t_predict

    print('Prediction took %f seconds using GPUs %s' % str(gpu))
    #with open(os.path.join(out_file, 't-inf_gpu%i.txt' % gpu), 'w') as f:
    #    f.write("Inference with gpu %i in %f s" % (gpu, t_predict))
    return True


if __name__ == '__main__':
    sample = sys.argv[1]
    gpu = int(sys.argv[2])
    single_gpu_inference(sample, gpu)
