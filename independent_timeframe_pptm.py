#!/usr/bin/env python

"""independent_timeframe_pptm.py: A postprocessing template for mutually independent time frames.
   Example usuage:
    # load your data and define the operations for each time frame
    # note that you don't need to load the data yourself, just specify the file location as keyword parameter and
    # the names of the function and its keywords are arbitrary but have to be consistent
    kwdict = {'data1':'/path/to/directory1', 'data2':'/path/to/directory2'}
    def postprocess(data1, data2):
        res = fft(data1) + data2
        save(res, 'res')
        return average(res, dim=1)  # or return nothing at all
    # then simply launch it, MPI parallelism will be invoked if possible
    launch(postprocess, kwdict[, outdir='/path/to/save/'])

    # optionally you can collect the result of each timeframe. you will get comm, rank, size and total_time to work with
    # you will also get an array of results from each node as input:
    def aggr(lst):
        r = comm.gather(lst, root=0)
        # do something to r
    launch(postprocess, kwdict, afunc=aggr)
"""

__author__ = "Han Wen"
__copyright__ = "Copyright 2018, PICKSC"
__credits__ = ["Adam Tableman", "Frank Tsung", "Thamine Dalichaouch"]
__license__ = "GPLv2"
__version__ = "0.1"
__maintainer__ = "Han Wen"
__email__ = "hanwen@ucla.edu"
__status__ = "Development"

import os
import glob
import numpy as np
import traceback
from itertools import chain
from osh5io import read_h5, write_h5
try:
    # importing mpi4py cause hard crash on some login nodes, use the following flag to disable mpi4py
    if not os.environ.get('IGNORE_MPI4PY_IMPORT'):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        raise ImportError
except ImportError:
    comm, rank, size = None, 0, 1
total_time = 0


def save(sd, dataset_name):
    return save_funchook(sd, dataset_name)


# define a few commonly used MPI calls for collecting final results
# we separate the collecting method from filler function to final container. user can freely combine the two
def mpi_gather(towhat):
    def collector(sublist):
        if not comm:
            return sublist
        lst = comm.gather(sublist, root=0)
        if rank == 0:
            return towhat(lst)
    return collector


def mpi_allgather(towhat):
    def collector(sublist):
        if not comm:
            return sublist
        lst = comm.allgather(sublist)
        return towhat(lst)
    return collector


# a few examples
@mpi_gather
def gather2list_iter(lst):
    return chain.from_iterable(lst)  # this will flatten the list, e. g. [[1,2],[3]] -> [1,2,3]


@mpi_gather
def gather2list(lst):
    return list(chain.from_iterable(lst))


@mpi_allgather
def allgather2list(lst):
    return [r for sl in lst for r in sl]  # another way to flatten the list, 2 for-loops that read backwards


@mpi_gather
def gather_to_zipped_tuple_of_lists_group_by_quantity(lst):  # e.g. [ [(,), (,)], [(,)] ] -> /[, , ,], [, , ,]/
    return zip(*chain.from_iterable(lst))


def sum2var(sublist):
    subtotal, total = 0, 0
    for v in sublist:
        subtotal += v
    comm.Reduce(subtotal, total, op=MPI.SUM, root=0)
    return total


def launch(func, kw4func, outdir=None, afunc=None):
    """wrap MPI calls & for loops around user defined postprocessing function"""
    # define the save function and make it global. we need to define it here to get the outdir info from launch()
    global save_funchook

    def save_funchook(sd, dataset_name):
        odir = './PPR/' if not outdir else outdir
        odir += '/' + dataset_name + '/'
        # can't add barrier here due to uneven work load
        if rank == 0:
            if not os.path.exists(odir):  # prepare output dir
                os.makedirs(odir)
        # print('rank ' + str(rank) + 'writng to '+ odir)
        write_h5(sd, path=odir, dataset_name=dataset_name)

    fdict, sdict, fnum, kwargs, sfr = {}, {}, [], {}, []
    if rank == 0:
        for k, v in kw4func.items():
            if isinstance(v, str):  # string is treated as
                if os.path.isfile(v):  # files
                    sdict[k] = v
                else:                  # or as path
                    fdict[k] = sorted(glob.glob(v + '/*.h5'))
                    fnum.append(len(fdict[k]))
                    if fnum[-1] == 0:
                        raise IOError('No h5 files found in ' + v)
            else:
                kwargs[k] = v

        if fnum.count(fnum[0]) != len(fnum):
            raise Exception('Number of files must be the same for all directories')
        # TODO(2) we should check if all quantities have exactly the same timestamp
    if comm:
        [fdict, sdict, kwargs, fnum] = comm.bcast([fdict, sdict, kwargs, fnum], root=0)
    # # divide the task
    global total_time
    total_time = fnum[0]
    my_share = (total_time - 1) // size + 1
    i_begin = rank * my_share
    i_end = (rank + 1) * my_share
    if i_end > total_time:
        i_end = total_time
    # rank0 loop once to setup necessary dirs. not pretty but solve the racing condition
    if comm:
        try:
            if rank == 0:
                i_begin = 1
                for k in fdict:
                    kwargs[k] = read_h5(fdict[k][0])
                sfr.append(func(**kwargs))  # store results for final aggregation
            comm.Barrier()
        except:
            print(traceback.format_exc(), flush=True)
            comm.Abort(errorcode=1)

    # load static files
    try:
        for k, v in sdict.items():
            sdict[k] = read_h5(v)
    except:
        print(traceback.format_exc(), flush=True)
        if comm:
            comm.Abort(errorcode=2)

    kwargs.update(sdict)  # here are all static parameters

    for i in range(i_begin, i_end):
        for k in fdict:
            kwargs[k] = read_h5(fdict[k][i])
        sfr.append(func(**kwargs))  # store results for final aggregation

    # it is up to the users to decide how to aggregate the results
    if afunc:
        try:
            afunc(sfr)
        except:
            print(traceback.format_exc(), flush=True)
            if comm:
                comm.Abort(errorcode=3)

