from __future__ import print_function
import os
import sys
import z5py
from dask import delayed, compute, threaded
from simpleference.inference.util import get_offset_lists

from run_inference import single_gpu_inference


def complete_inference(sample, gpu_list):

    # path to the raw data
    raw_path = ('/groups/saalfeld/home/papec/Work/neurodata_hdd/'
                'cremi_warped/sample%s.n5' % sample)
    rf = z5py.File(raw_path, use_zarr_format=False)
    shape = rf['raw'].shape

    # create the datasets
    out_shape = (56,) * 3
    out_file = ('/groups/saalfeld/home/nuneziglesiasj/data/simpleference/'
                'torch_master_test_sample%s.n5' % sample)

    # the n5 file might exist already
    if not os.path.exists(out_file):
        f = z5py.File(out_file, use_zarr_format=False)
        f.create_dataset('affs_xy', shape=shape,
                         compressor='gzip',
                         dtype='float32',
                         chunks=out_shape)
        f.create_dataset('affs_z', shape=shape,
                         compressor='gzip',
                         dtype='float32',
                         chunks=out_shape)

    # make the offset files, that assign blocks to gpus
    save_folder = ('/groups/saalfeld/home/nuneziglesiasj/data/simpleference/'
                   'offsets_sample%s' % sample)
    output_shape = (56, 56, 56)
    get_offset_lists(shape, gpu_list, save_folder, output_shape=output_shape)

    tasks = [delayed(single_gpu_inference)(sample, gpu) for gpu in gpu_list]
    result = compute(*tasks, traverse=False,
                     get=threaded.get, num_workers=len(gpu_list))

    if all(result):
        print("All gpu's finished inference properly.")
    else:
        print("WARNING: at least one process didn't finish properly.")


if __name__ == '__main__':
    gpu = int(sys.argv[1])
    for sample in ('A+',):
        gpu_list = [gpu]
        complete_inference(sample, gpu_list)
