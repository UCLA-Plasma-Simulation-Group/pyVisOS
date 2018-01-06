#!/usr/bin/env python

"""independent_timeframe_pptm.py: A postprocessing template for mutually independent time frames.
   Example usuage:
    # load your data and define the operations for each time frame
    # note that you don't need to load the data yourself, just specify the file location as keyword parameter and
    # the names of function and the its keywords are arbitrary
    def postprocess(data1='/path/to/directory1', data2='/path/to/directory2'[, data3=...]):
        res = fft(data1) + data2
        save(res, 'res')
    # then simply launch it, MPI parallelism will be invoked if possible
    launch(postprocess[, outdir='/path/to/save/'])
"""

__author__ = "Han Wen"
__copyright__ = "Copyright 2018, The Cogent Project"
__credits__ = ["Adam Tableman", "Frank Tsung", "Thamine Dalichaouch"]
__license__ = "GPLv2"
__version__ = "0.1"
__maintainer__ = "Han Wen"
__email__ = "hanwen@ucla.edu"
__status__ = "Development"

import inspect
import os
import glob
from osh5io import read_hdf, write_hdf


def launch(func, outdir=None):
    """wrap MPI calls & for loops around user defined postprocessing function"""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except ImportError:
        comm, rank, size = None, 0, 1

    # define the save function and make it global. we need to define it here to get the outdir info from launch
    global save

    def save(sd, dataset_name):
        odir = './' if not outdir else outdir
        odir += '/PPR/' + dataset_name + '/'
        # TODO(1) there is a racing condition. rank>0 nodes may write to a non-existing dir. EDIT: solved
        # can't add barrier here due to uneven work load
        if rank == 0:
            if not os.path.exists(odir):  # prepare output dir
                os.makedirs(odir)
        # print('rank ' + str(rank) + 'writng to '+ odir)
        write_hdf(sd, path=odir, dataset_name=dataset_name)

    # get function signature so that we know what files to load
    sig = inspect.signature(func)

    fdict, fnum, kwargs = {}, [], {}
    if rank == 0:
        for k, v in sig.parameters.items():
            fdict[k] = sorted(glob.glob(v.default + '/*.h5'))
            fnum.append(len(fdict[k]))
            if fnum[-1] == 0:
                raise IOError('No h5 files found in' + v.default)

        if fnum.count(fnum[0]) != len(fnum):
            raise Exception('Number of files must be the same for all directories')
        # TODO(2) we should check if all quantities have exactly the same timestamp
    if comm:
        [fdict, fnum] = comm.bcast([fdict, fnum], root=0)
    # # divide the task
    total_time = fnum[0]
    my_share = (total_time - 1) // size + 1
    i_begin = rank * my_share
    i_end = (rank + 1) * my_share
    if i_end > total_time:
        i_end = total_time
    # rank0 loop once to setup necessary dirs. not pretty but solve TODO(1)
    if comm:
        if rank == 0:
            i_begin = 1
            for k in fdict:
                kwargs[k] = read_hdf(fdict[k][0])
            func(**kwargs)
        comm.Barrier()

    for i in range(i_begin, i_end):
        for k in fdict:
            kwargs[k] = read_hdf(fdict[k][i])
        func(**kwargs)

