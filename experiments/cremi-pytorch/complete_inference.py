from __future__ import print_function
import yaml
import z5py
from dask import delayed, compute, threaded
from simpleference.inference.util import get_offset_lists

from run_inference import single_gpu_inference


def complete_inference(config_file):

    # load parameters from JSON config file
    with open(config_file, 'r') as conf:
        config = yaml.load(conf)

    sample = config['sample']

    # path to the raw data
    raw_path = config['raw_path']
    rf = z5py.File(raw_path, use_zarr_format=False)
    shape = rf['raw'].shape

    # create the datasets
    out_shape = config['out_shape']
    out_file = config['out_file']

    # the n5 file might exist already
    f = z5py.File(out_file, use_zarr_format=False)
    if 'affs_xy' not in f:
        f.create_dataset('affs_xy', shape=shape,
                         compression='gzip',
                         dtype='float32',
                         chunks=out_shape)
    if 'affs_z' not in f:
        f.create_dataset('affs_z', shape=shape,
                         compression='gzip',
                         dtype='float32',
                         chunks=out_shape)

    # make the offset files, that assign blocks to gpus
    save_folder = config['save_folder']
    output_shape = config['output_shape']
    gpu_list = config['gpu_list']
    get_offset_lists(shape, gpu_list, save_folder, output_shape=output_shape)

    tasks = [delayed(single_gpu_inference)(sample, gpu) for gpu in gpu_list]
    result = compute(*tasks, traverse=False,
                     get=threaded.get, num_workers=len(gpu_list))

    if all(result):
        print("All gpu's finished inference properly.")
    else:
        print("WARNING: at least one process didn't finish properly.")


if __name__ == '__main__':
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    complete_inference(config_file)
