#!/usr/bin/env python

"""osaxis.py: Disk IO for the OSIRIS HDF5 data."""

__author__ = "Han Wen"
__copyright__ = "Copyright 2018, PICKSC"
__credits__ = ["Adam Tableman", "Frank Tsung", "Thamine Dalichaouch"]
__license__ = "GPLv2"
__version__ = "0.1"
__maintainer__ = "Han Wen"
__email__ = "hanwen@ucla.edu"
__status__ = "Development"


import h5py
import os
from osh5def import *
from osunit import *


def read_hdf(filename, path=None):
    """
    HDF reader for Osiris/Visxd compatible HDF files... This will slurp in the data
    and the attributes that describe the data (e.g. title, units, scale).

    Usage:
            diag_data = read_hdf('e1-000006.h5')

            data = diag_data.data                         # gets the raw data
            print(diag_data.data.shape)                   # prints the dimension of the raw data
            print(diag_data.run_attrs['TIME'])       # prints the simulation time associated with the hdf5 file
            diag_data.data_attrs['UNITS']             # print units of the dataset points
            list(diag_data.data_attrs)             # lists all attributes related to the data array
            list(diag_data.run_attrs)                 # lists all attributes related to the run
            print diag_data.axes[0].attrs['UNITS']    # prints units of  X-axis
            list(diag_data.axes[0].attrs['UNITS'])    # lists all variables of the X-axis

            diag_data.slice( x=34, y=(10,30) )
            diag_data.slice(x=3)

            diag_data.write(diag_data, 'filename.h5')    # writes out Visxd compatible HDF5 data.

    """
    fname = filename if not path else path + '/' + filename
    data_file = h5py.File(fname, 'r')

    the_data_hdf_object = scan_hdf5_file_for_main_data_array(data_file)

    data_bundle = HDFData()
    data_bundle.timestamp = HDFData.fn_rule.findall(os.path.basename(filename))
    data_bundle.name = the_data_hdf_object.name[1:]  # ignore the beginning '/'

    # now read in attributes of the ROOT of the hdf5..
    #   there's lots of good info there.
    for key, value in data_file.attrs.items():
        data_bundle.run_attrs[key] = value

    # attach attributes assigned to the data array to
    #    the HDFData.data_attrs object
    for key, value in the_data_hdf_object.attrs.items():
        data_bundle.data_attrs[key] = value

    # convert unit string to osunit object
    try:
        data_bundle.data_attrs['UNITS'] = np.array([OSUnits(data_bundle.data_attrs['UNITS'][0])])
    except KeyError:
        data_bundle.data_attrs['UNITS'] = np.array([OSUnits('a.u.')])
    data_bundle.sim_dim = np.size(data_bundle.run_attrs['XMAX'])

    axis_number = 1
    while True:
        try:
            # try to open up another AXIS object in the HDF's attribute directory
            #  (they are named /AXIS/AXIS1, /AXIS/AXIS2, /AXIS/AXIS3 ...)
            axis_to_look_for = "/AXIS/AXIS" + str(axis_number)
            axis = data_file[axis_to_look_for]
            axis_min = axis[0]
            axis_max = axis[1]
            axis_numberpoints = the_data_hdf_object.shape[-axis_number]

            data_axis = DataAxis(axis_min, axis_max, axis_numberpoints, attrs=axis.attrs)
            data_bundle.axes.insert(0, data_axis)
        except KeyError:
            break
        axis_number += 1

    data_bundle.data = the_data_hdf_object[()]

    data_file.close()
    return data_bundle


def scan_hdf5_file_for_main_data_array(h5file):
    for k, v in h5file.items():
        if isinstance(v, h5py.Dataset):
            return h5file[k]
    else:
        raise Exception('Main data array not found')


def write_hdf(data, filename=None, path=None, dataset_name=None, write_data=True):
    if isinstance(data, HDFData):
        data_object = data
    elif isinstance(data, np.ndarray):
        data_object = HDFData()
        data_object.data = data
        data_object.sim_dim = 0  # we don't know the simulation dimension.
    else:
        try:  # maybe it's something we can wrap in a numpy array
            data = np.array(data)
            data_object = HDFData()
            data_object.data = data
            data_object.sim_dim = np.ndim(data)
        except:
            raise Exception(
                "Invalid data type.. we need a 'hdf5_data', numpy array, or somehitng that can go in a numy array")

    # now let's make the HDFData() compatible with VisXd and such...
    # take care of the NAME attribute.
    if dataset_name is not None:
        current_name_attr = dataset_name
    elif data_object.name:
        current_name_attr = data_object.name
    else:
        current_name_attr = "Data"

    fname = path if path else ''
    if filename is not None:
        fname += filename
    elif data_object.timestamp is not None:
        fname += current_name_attr + '-' + data_object.timestamp + '.h5'
    else:
        raise Exception("You did not specify a filename!!!")
    if os.path.isfile(fname):
        os.remove(fname)
    h5file = h5py.File(fname)

    # now put the data in a group called this...
    h5dataset = h5file.create_dataset(current_name_attr, data_object.data.shape, data=data_object.data)
    # these are required.. so make defaults ones...
    h5dataset.attrs['UNITS'], h5dataset.attrs['LONG_NAME'] = b'', b''
    # convert osunit class back to ascii
    try:
        data_object.data_attrs['UNITS'] = np.array([str(data_object.data_attrs['UNITS'][0]).encode('utf-8')])
    except:
        data_object.data_attrs['UNITS'] = np.array([b'a.u.'])
    # copy over any values we have in the 'HDFData' object;
    for key, value in data_object.data_attrs.items():
        h5dataset.attrs[key] = value
    # these are required so we make defaults..
    h5file.attrs['DT'] = 1.0
    h5file.attrs['ITER'] = 0
    h5file.attrs['MOVE C'] = [0] * data_object.sim_dim
    h5file.attrs['PERIODIC'] = [0] * data_object.sim_dim
    h5file.attrs['TIME'] = 0.0
    h5file.attrs['TIME UNITS'] = ''
    h5file.attrs['TYPE'] = b'grid'
    h5file.attrs['XMIN'] = [0.0] * data_object.sim_dim
    h5file.attrs['XMAX'] = [1.0] * data_object.sim_dim
    # now make defaults/copy over the attributes in the root of the hdf5
    for key, value in data_object.run_attrs.items():
        h5file.attrs[key] = value

    number_axis_objects_we_need = len(data_object.axes)
    # now go through and set/create our axes HDF entries.
    for i in range(0, number_axis_objects_we_need):
        axis_name = "AXIS/AXIS%d" % (number_axis_objects_we_need - i)
        if axis_name not in h5file:
            axis_data = h5file.create_dataset(axis_name, (2,), 'float64')
        else:
            axis_data = h5file[axis_data]

        # set the extent to the data we have...
        axis_data[0] = data_object.axes[i].axis_min()
        axis_data[1] = data_object.axes[i].axis_max()

        # now make attributes for axis that are required..
        axis_data.attrs['UNITS'], axis_data.attrs['LONG_NAME'], axis_data.attrs['TYPE'], axis_data.attrs['NAME'] = \
            b"", b"", b"", b""
        # fill in any values we have stored in the Axis object
        for key, value in data_object.axes[i].attrs.items():
            axis_data.attrs[key] = value
    if write_data:
        h5file.close()


if __name__ == '__main__':
    b=read_hdf('/u/home/h/hwen/oldscratch/extdrivenepw-morepart/c6e-6/MS/PHA/p1x1/electrons/p1x1-electrons-001348.h5')
    a = b.data_attrs['UNITS'][0].tex()
    write_hdf(b, '/u/home/h/hwen/scratch/tmp/p1x1-electrons-001348.h5')
    c=read_hdf('/u/home/h/hwen/scratch/tmp/p1x1-electrons-001348.h5')
    print(type(c.data_attrs['UNITS'][0]))
    print(c.data_attrs['UNITS'][0])
