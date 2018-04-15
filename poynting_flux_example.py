#!/usr/bin/env python


"""Calculate poyning flux for each timestamp and then average over transverse direction.
    Finally we collect all the data and combine the 1d results into a 2d array with (x,t) as coordinates'
"""
import independent_timeframe_pptm as itp  # you need to import this one
import osh5utils  # provide useful functions to manipulate H5Data
import os  # if you need to get current working path
import osh5vis  # if you want to plot something

# Specify what to load in a dictionary. Note that we only need to write down the directories
# Note that all string values are reserved for the filename and path. If you need to pass in
# strings, you can either wrap them into list/tuple/dict or use bytes like b'parameters'.
__pwd = os.getcwd()
__ldq = {'e2': __pwd + '/MS/FLD/e2-senv',
           'e3': __pwd + '/MS/FLD/e3-senv',
           'b2': __pwd + '/MS/FLD/b2-senv',
           'b3': __pwd + '/MS/FLD/b3-senv'}


# Note that the names must be the same as the keys in the dictionary. However the order is not important.
def calculate_poyning_flux(b2, b3, e2, e3, save2disk=True):
    # inside the function we can treat each arguments as individual files instead of directories
    # what happens is that the itp wrapper function will distribute the workload along timestamp axis and load the
    # files into your named keywords one timestamp at a time. You should now see e2, for example, as an
    # H5Data representing e2-senv-123456.h5.

    # basic operators have been overloaded
    s1 = e2 * b3 - e3 * b2  # poynting vector long x1
    # let's rename the variable
    s1.name = s1.data_attrs['NAME'] = 's1'
    s1.data_attrs['LONG_NAME'] = 's_1'

    # you can slice the data and the axis will follow automatically
    nx = s1.axes[1].size
    s1 = s1[..., nx//20: -nx//20-1]

    # integrate over x2 (and x3 if it is a 3D sim.), s1 will be 1D now.
    axis2sum = 0 if s1.ndim == 2 else (1, 0)
    s1 = s1.sum(axis=axis2sum)

    if save2disk:
        itp.save(s1, 's1')  # not necessary, just to show that you can save intermediate results
    return s1   # optional, all returns will be wrapped into a list


# This function is optional in general. Here we need it to combine the 1D output to a 2D figure
# Its first argument has to be a list. You can use 3 global values:
# comm: MPI communicator; rank: rank of the MPI process; size: number of MPI processes
def combine2fig(subl):
    # all MPI processes will excute the code unless specified otherwise

    # head node will get the full list, others get None. See independent_timeframe_pptm.py for detail
    full_list = itp.gather2list(subl)
    if itp.rank == 0:
        data = osh5utils.stack(full_list)  # similar to numpy.stack, the new dimension is assumed to be time by default
        osh5vis.new_fig(data.transpose())  # create a new plot of transposed data, and display


# Now we need to run it
# Keyword arguments are optional.
# outdir specifies the root output dir, default is ./PPR/. In our case the output will be stored at ./test/s1/
itp.launch(calculate_poyning_flux, __ldq, outdir='./test', afunc=combine2fig)

# Suppose you also want to use the full fields instead of the enveloped fields. You can
__ldq = {'e2': __pwd + '/MS/FLD/e2', 'e3': __pwd + '/MS/FLD/e3',
           'b2': __pwd + '/MS/FLD/b2', 'b3': __pwd + '/MS/FLD/b3',
           'save2disk': False}
itp.launch(calculate_poyning_flux, __ldq, afunc=combine2fig)
