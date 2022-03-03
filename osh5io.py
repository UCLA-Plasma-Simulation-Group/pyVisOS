#!/usr/bin/env python

"""
osh5io.py
=========
Disk IO for the OSIRIS HDF5 data.
"""

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
import numpy as np
from osh5def import H5Data, PartData, fn_rule, DataAxis, OSUnits
try:
    import zdf

    def read_zdf(filename, path=None):
        """
        HDF reader for Osiris/Visxd compatible HDF files.
        Returns: H5Data object.
        """
        fname = filename if not path else path + '/' + filename
        data, info = zdf.read(filename)
        run_attrs, data_attrs = {}, {}
        nx = list(reversed(info.grid.nx))
        axes = [DataAxis(ax.min, ax.max, nx[i],
                         attrs={'LONG_NAME': ax.label,
                                'NAME': ax.label.replace('_', ''),
                                'UNITS': OSUnits(ax.units)})
                for i, ax in enumerate(reversed(info.grid.axis))]

        timestamp=fn_rule.findall(os.path.basename(filename))[0]

        run_attrs['NX']=info.grid.nx
        run_attrs['TIME UNITS'] = OSUnits(info.iteration.tunits)
        run_attrs['TIME']=np.array([info.iteration.t])
        run_attrs['TIMESTAMP']=timestamp

        data_attrs['LONG_NAME']=info.grid.label
        data_attrs['NAME']=info.grid.label.replace('_', '')
        data_attrs['UNITS']=OSUnits(info.grid.units)
        return H5Data(data, timestamp=timestamp, data_attrs=data_attrs, run_attrs=run_attrs, axes=axes)

except ImportError:
    def read_zdf(_):
        raise NotImplementedError('Cannot import zdf reader, zdf format not supported')


def read_grid(filename, path=None, axis_name="AXIS/AXIS"):
    """
    Read grid data from Osiris/OSHUN output. Data can be in hdf5 or zdf format
    """
    ext = os.path.basename(filename).split(sep='.')[-1]

    if ext == 'h5':
        return read_h5(filename, path=path, axis_name="AXIS/AXIS")
    elif ext == 'zdf':
        return read_zdf(filename, path=path)
    else:
        # the file extension may not be specified, trying all supported formats
        try:
            return read_h5(filename+'.h5', path=path, axis_name="AXIS/AXIS")
        except OSError:
            return read_zdf(filename+'.zdf', path=path)


def read_h5(filename, path=None, axis_name="AXIS/AXIS"):
    """
    HDF reader for Osiris/Visxd compatible HDF files... This will slurp in the data
    and the attributes that describe the data (e.g. title, units, scale).

    Usage:
            diag_data = read_hdf('e1-000006.h5')      # diag_data is a subclass of numpy.ndarray with extra attributes

            print(diag_data)                          # print the meta data
            print(diag_data.view(numpy.ndarray))      # print the raw data
            print(diag_data.shape)                    # prints the dimension of the raw data
            print(diag_data.run_attrs['TIME'])        # prints the simulation time associated with the hdf5 file
            diag_data.data_attrs['UNITS']             # print units of the dataset points
            list(diag_data.data_attrs)                # lists all attributes related to the data array
            list(diag_data.run_attrs)                 # lists all attributes related to the run
            print(diag_data.axes[0].attrs['UNITS'])   # prints units of X-axis
            list(diag_data.axes[0].attrs)             # lists all variables of the X-axis

            diag_data[slice(3)]
                print(rw.view(np.ndarray))

    We will convert all byte strings stored in the h5 file to strings which are easier to deal with when writing codes
    see also write_h5() function in this file

    """
    fname = filename if not path else path + '/' + filename
    data_file = h5py.File(fname, 'r')

    n_data = scan_hdf5_file_for_main_data_array(data_file)

    timestamp, name, run_attrs, data_attrs, axes, data_bundle= '', '', {}, {}, [], []
    try:
        timestamp = fn_rule.findall(os.path.basename(filename))[0]
    except IndexError:
        timestamp = '000000'

    axis_number = 1
    while True:
        try:
            # try to open up another AXIS object in the HDF's attribute directory
            #  (they are named /AXIS/AXIS1, /AXIS/AXIS2, /AXIS/AXIS3 ...)
            axis_to_look_for = axis_name + str(axis_number)
            axis = data_file[axis_to_look_for]
            # convert byte string attributes to string
            attrs = {}
            for k, v in axis.attrs.items():
                try:
                    attrs[k] = v[0].decode('utf-8') if isinstance(v[0], bytes) else v
                except IndexError:
                    attrs[k] = v.decode('utf-8') if isinstance(v, bytes) else v

            axis_min = axis[0]
            axis_max = axis[-1]
            axis_numberpoints = n_data[0].shape[-axis_number]

            data_axis = DataAxis(axis_min, axis_max, axis_numberpoints, attrs=attrs)
            axes.insert(0, data_axis)
        except KeyError:
            break
        axis_number += 1

    # we need a loop here primarily (I think) for n_ene_bin phasespace data
    for the_data_hdf_object in n_data:
        name = the_data_hdf_object.name[1:]  # ignore the beginning '/'

        # now read in attributes of the ROOT of the hdf5..
        #   there's lots of good info there. strip out the array if value is a string

        for key, value in data_file.attrs.items():
            try:
                run_attrs[key] = value[0].decode('utf-8') if isinstance(value[0], bytes) else value
            except IndexError:
                run_attrs[key] = value.decode('utf-8') if isinstance(value, bytes) else value
        try:
            run_attrs['TIME UNITS'] = OSUnits(run_attrs['TIME UNITS'])
        except:
            run_attrs['TIME UNITS'] = OSUnits('1 / \omega_p')
        # attach attributes assigned to the data array to
        #    the H5Data.data_attrs object, remove trivial dimension before assignment
        for key, value in the_data_hdf_object.attrs.items():
            try:
                data_attrs[key] = value[0].decode('utf-8') if isinstance(value[0], bytes) else value
            except IndexError:
                data_attrs[key] = value.decode('utf-8') if isinstance(value, bytes) else value

        # check if new data format is in use
        if not data_attrs and 'SIMULATION' in data_file:
            data_attrs['LONG_NAME'], data_attrs['UNITS'] = run_attrs.pop('LABEL', 'data'), run_attrs.pop('UNITS', OSUnits('a.u.'))
            run_attrs['SIMULATION'] = {k:v for k,v in data_file['/SIMULATION'].attrs.items()}
        # convert unit string to osunit object
        try:
            data_attrs['UNITS'] = OSUnits(data_attrs['UNITS'])
        except:
#             data_attrs['UNITS'] = OSUnits('a.u.')
            pass
        data_attrs['NAME'] = name

        # data_bundle.data = the_data_hdf_object[()]
        data_bundle.append(H5Data(the_data_hdf_object, timestamp=timestamp,
                                  data_attrs=data_attrs, run_attrs=run_attrs, axes=axes))
    data_file.close()
    if len(data_bundle) == 1:
        return data_bundle[0]
    else:
        return data_bundle


def read_raw(filename, path=None):
    """
    Read particle raw data into a numpy sturctured array.
    See numpy documents for detailed usage examples of the structured array.
    The only modification is that the meta data of the particles are stored in .attrs attributes.
    
    Usage:
            part = read_raw("raw-electron-000000.h5")   # part is a subclass of numpy.ndarray with extra attributes
            
            print(part.shape)                           # should be a 1D array with # of particles
            print(part.attrs)                           # print all the meta data
            print(part.attrs['TIME'])                   # prints the simulation time associated with the hdf5 file
    """
    fname = filename if not path else path + '/' + filename
    try:
        timestamp = fn_rule.findall(os.path.basename(filename))[0]
    except IndexError:
        timestamp = '000000'
    with h5py.File(fname, 'r') as data:
        quants = [k for k in data.keys()]
        new_ver = 'SIMULATION' in quants
        if new_ver:
            quants.remove('SIMULATION')

        # read in meta data
        d = {k:v for k,v in data.attrs.items()}
        # in the old version label and units are stored inside each quantity dataset
        if not new_ver:
            d['LABELS'] = [data[q].attrs['LONG_NAME'][0].decode() for q in quants]
            d['UNITS'] = [data[q].attrs['UNITS'][0].decode() for q in quants]
        else:
            d.update({k:v for k, v in data['SIMULATION'].attrs.items()})
            d['LABELS'] = [n.decode() for n in d['LABELS']]
            d['UNITS'] = [n.decode() for n in d['UNITS']]
        d['QUANTS'] = quants
        #TODO: TIMESTAMP is not set in HDF5 file as of now (Aug 2019) so we make one up, check back when file format changes
        d['TIMESTAMP'] = timestamp

        dtype = [(q, data[q].dtype) for q in quants]
        r = PartData(data[dtype[0][0]].shape, dtype=dtype, attrs=d)
        for dt in dtype:
            r[dt[0]] = data[dt[0]]

    return r


def read_h5_openpmd(filename, path=None):
    """
    HDF reader for OpenPMD compatible HDF files... This will slurp in the data
    and the attributes that describe the data (e.g. title, units, scale).

    Usage:
            diag_data = read_hdf_openpmd('EandB000006.h5')      # diag_data is a subclass of numpy.ndarray with extra attributes

            print(diag_data)                          # print the meta data
            print(diag_data.view(numpy.ndarray))      # print the raw data
            print(diag_data.shape)                    # prints the dimension of the raw data
            print(diag_data.run_attrs['TIME'])        # prints the simulation time associated with the hdf5 file
            diag_data.data_attrs['UNITS']             # print units of the dataset points
            list(diag_data.data_attrs)                # lists all attributes related to the data array
            list(diag_data.run_attrs)                 # lists all attributes related to the run
            print(diag_data.axes[0].attrs['UNITS'])   # prints units of X-axis
            list(diag_data.axes[0].attrs)             # lists all variables of the X-axis

            diag_data[slice(3)]
                print(rw.view(np.ndarray))

    We will convert all byte strings stored in the h5 file to strings which are easier to deal with when writing codes
    see also write_h5() function in this file

    """
    fname = filename if not path else path + '/' + filename
    try:
        timestamp = fn_rule.findall(os.path.basename(filename))[0]
    except IndexError:
        pass
    with h5py.File(fname, 'r') as data_file:
        data_file = h5py.File(filename, 'r')
        attrs_f = {k:v for k,v in data_file.attrs.items()}
        bp = attrs_f.pop('basePath', b'/data/%T/')
        dataPath = bp.replace(b'/%T', b'').decode()
        mp = attrs_f.pop('meshesPath', b'').decode()
        pp = attrs_f.pop('particlesPath', b'').decode()

        lname_dict, fldl = {'E1': 'E_x', 'E2': 'E_y', 'E3': 'E_z',
                            'B1': 'B_x', 'B2': 'B_y', 'B3': 'B_z',
                            'Ex': 'E_x', 'Ey': 'E_y', 'Ez': 'E_z',
                            'Bx': 'B_x', 'By': 'B_y', 'Bz': 'B_z',
                            'jx': 'J_x', 'jy': 'J_y', 'jz': 'J_z', 'rho': r'\rho'}, {}
        for it, data_group in data_file[dataPath].items():
            basePath = dataPath+it
            base_attrs = {k.upper():v for k,v in data_file[basePath].attrs.items()}
            base_attrs['TIME UNITS'] = OSUnits('')
            base_attrs.update(attrs_f)
            timestamp = '%07i'%int(it)
            # get grid quantities
            meshesPath = basePath + '/' + mp
            # get general grid quantity attributes
            grid_attrs = {k.upper():v for k,v in data_file[meshesPath].attrs.items()}
            grid_attrs.update(base_attrs)
            for fld, data in data_file[meshesPath].items():
                data_attrs = {k:v for k, v in data.attrs.items()}
                axisLabels = data_attrs.pop('axisLabels')
                gridSpacing = data_attrs.pop('gridSpacing')
                gridGlobalOffset = data_attrs.pop('gridGlobalOffset', 0)
                gridUnitSI = data_attrs.pop('gridUnitSI', 1.0)
                #TODO: make sure the axis always corresponds to real space (is it true? where else can we get the axis unitDimension?)
                unitDimension = data_attrs.pop('unitDimension', np.array([0.,0.,0.,0.,0.,0.,0.]))
                grid_units, g_fac = __convert_to_osiris_units(np.array([1.,0.,0.,0.,0.,0.,0.]), gridUnitSI)
                gridUnitSI *= g_fac
                unitSI = data_attrs.pop('unitSI', 1.0)
                data_units, d_fac =  __convert_to_osiris_units(unitDimension, unitSI)
                unitSI *= d_fac
                # scalar data
                if isinstance(data, h5py.Dataset):
                    position = data_attrs.pop('position', 0)
                    axis_min, axis_max = __get_openPMD_dataaxis_limits(gridGlobalOffset, position, gridSpacing, gridUnitSI, data.shape)
                    data_attrs.update( {'LONG_NAME': lname_dict.get(fld, fld), 'NAME': fld} )
                    axes = __generate_dataaxis(axisLabels, axis_max, axis_min, grid_units,
                                               data.shape, data_attrs['dataOrder'].decode())
                    fldl[it+'/'+data_attrs['NAME']] = (H5Data(data, timestamp=timestamp, data_attrs=data_attrs,
                                                              run_attrs=grid_attrs, axes=axes))
                # vector data
                elif isinstance(data, h5py.Group):
                    for k, v in data.items():
                        comp_attrs = {kk:vv for kk,vv in v.attrs.items()}
                        position = comp_attrs.pop('position', 0)
                        data_attrs.update(comp_attrs)
                        axis_min, axis_max = __get_openPMD_dataaxis_limits(gridGlobalOffset, position, gridSpacing, gridUnitSI, v.shape)
                        data_attrs.update( {'LONG_NAME': lname_dict.get(fld+k, fld+'_'+k), 'NAME': fld+k} )
                        axes = __generate_dataaxis(axisLabels, axis_max, axis_min, grid_units,
                                                   v.shape, data_attrs['dataOrder'].decode())
                        fldl[it+'/'+data_attrs['NAME']] = (H5Data(v, timestamp=timestamp, data_attrs=data_attrs,
                                                                  run_attrs=grid_attrs, axes=axes))
            #TODO: read the particle data (the particle data format is not settled in h5Data, so just return the data group ATM)
            # get particle data
            particlePath = basePath + '/' + pp
            part_attrs = {k.upper():v for k,v in data_file[particlePath].attrs.items()}
            part_attrs.update(base_attrs)
            for spe, data in data_file[particlePath].items():
                fldl[it+'/'+spe] = data
        #
        #         print(spe, ':', data)
        #         for k, v in data.attrs.items():
        #             print(k, v)
        #         print('-----')
        #         for q, d in data.items():
        #             print(q, d)
    return fldl

def __convert_to_osiris_units(openPMDunit, unitSI):
    #TODO: convert openPMD unitDimenion to OSUnits
    ...
    return OSUnits(''), 1


def __get_openPMD_dataaxis_limits(gridGlobalOffset, position, gridSpacing, gridUnitSI, data_shape):
    axis_min = (gridGlobalOffset + position * gridSpacing) * gridUnitSI
    axis_max = (gridGlobalOffset + (position + data_shape) * gridSpacing) * gridUnitSI
    return axis_min, axis_max


def __generate_dataaxis(ax_label, ax_max, ax_min, ax_unit, data_shape, order):
    axes = []
    for an, amax, amin, anp in zip(ax_label, ax_max, ax_min, data_shape):
        ax_attrs = {'LONG_NAME': an.decode(), 'NAME': an.decode(), 'UNITS': ax_unit}
        data_axis = DataAxis(amin, amax, anp, attrs=ax_attrs)
        if order.upper() == 'F':
            axes.insert(0, data_axis)
        else:
            axes.append(data_axis)
    return axes


def __read_dataset_and_convert_to_h5data(k, v, data_attrs, dflt_ax_unit,
                                         timestamp, run_attrs):
    ax_label, ax_off, g_spacing, ax_pos, unitsi = \
        data_attrs.pop('axisLabels'), data_attrs.pop('gridGlobalOffset', 0), \
        data_attrs.pop('gridSpacing'), data_attrs.pop('position',
                                                      0), data_attrs.pop(
            'unitSI', 1.)
    ax_min = (ax_off + ax_pos * g_spacing) * unitsi
    ax_max = ax_min + v.shape * g_spacing * unitsi

    # prepare the axes data
    axes = []
    for aln, an, amax, amin, anp in zip(ax_label, ax_label,
                                        ax_max, ax_min, v.shape):
        ax_attrs = {'LONG_NAME': aln.decode('utf-8'),
                    'NAME': an.decode('utf-8'), 'UNITS': dflt_ax_unit}
        data_axis = DataAxis(amin, amax, anp, attrs=ax_attrs)
        axes.append(data_axis)
    return H5Data(v[()], timestamp=timestamp, data_attrs=data_attrs,
                  run_attrs=run_attrs, axes=axes)


def scan_hdf5_file_for_main_data_array(h5file):
    res = []
    for k, v in h5file.items():
        if isinstance(v, h5py.Dataset):
            res.append(h5file[k])
    if not res:
        raise Exception('Main data array not found')
    return res


def write_h5(data, filename=None, path=None, dataset_name=None, overwrite=True, axis_name=None):
    """
    Usage:
        write(diag_data, '/path/to/filename.h5')    # writes out Visxd compatible HDF5 data.

    Since h5 format does not support python strings, we will convert all string data (units, names etc)
    to bytes strings before writing.

    see also read_h5() function in this file

    """
    if isinstance(data, H5Data):
        data_object = data
    elif isinstance(data, np.ndarray):
        data_object = H5Data(data)
    else:
        try:  # maybe it's something we can wrap in a numpy array
            data_object = H5Data(np.array(data))
        except:
            raise Exception(
                "Invalid data type.. we need a 'hdf5_data', numpy array, or somehitng that can go in a numy array")

    # now let's make the H5Data() compatible with VisXd and such...
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
        if overwrite:
            os.remove(fname)
        else:
            c = 1
            while os.path.isfile(fname[:-3]+'.copy'+str(c)+'.h5'):
                c += 1
            fname = fname[:-3]+'.copy'+str(c)+'.h5'
    h5file = h5py.File(fname,'a')
    run_attrs = data_object.run_attrs.copy()

    # now put the data in a group called this...
    h5dataset = h5file.create_dataset(current_name_attr, data_object.shape, data=data_object.view(np.ndarray))

    # these are required so we make defaults..
    h5file.attrs['ITER'] = [0]
    h5file.attrs['TIME'] = [0.0]
    h5file.attrs['TIME UNITS'] = [b'1 / \omega_p']
    h5file.attrs['TYPE'] = [b'grid']
    # convert osunit class back to ascii
    data_attrs = data_object.data_attrs.copy()
    try:
        data_attrs['UNITS'] = np.array([str(data_object.data_attrs['UNITS']).encode('utf-8')])
    except:
        data_attrs['UNITS'] = np.array([b'a.u.'])

    # check if the data is read from a new format file
    new_format = run_attrs.pop('SIMULATION', False)
    if new_format:
        # now make defaults/copy over the attributes in the root of the hdf5
        for key, value in run_attrs.items():
            h5file.attrs[key] = np.array([value.encode('utf-8')]) if isinstance(value, (str, OSUnits)) else value
        h5file.attrs['LABEL'], h5file.attrs['UNITS'] = np.array([data_attrs['LONG_NAME'].encode('utf-8')]), data_attrs['UNITS']
        simgroup = h5file.create_group('SIMULATION')
        for k, v in new_format.items():
            simgroup.attrs[k] = v
    else:
        # complete the list of required defaults
        h5file.attrs['DT'] = [1.0]
        h5file.attrs['MOVE C'] = [0]
        h5file.attrs['PERIODIC'] = [0]
        h5file.attrs['XMIN'] = [0.0]
        h5file.attrs['XMAX'] = [0.0]
        # these are required.. so make defaults ones...
        h5dataset.attrs['UNITS'], h5dataset.attrs['LONG_NAME'] = np.array([b'']), np.array([b''])
        # copy over any values we have in the 'H5Data' object;
        for key, value in data_attrs.items():
            h5dataset.attrs[key] = np.array([value.encode('utf-8')]) if isinstance(value, str) else value
        # now make defaults/copy over the attributes in the root of the hdf5
        for key, value in run_attrs.items():
            h5file.attrs[key] = np.array([value.encode('utf-8')]) if isinstance(value, (str, OSUnits)) else value

    number_axis_objects_we_need = len(data_object.axes)
    # now go through and set/create our axes HDF entries.
    if not axis_name:
        axis_name = "AXIS/AXIS"
    for i in range(0, number_axis_objects_we_need):
        _axis_name = axis_name + str(number_axis_objects_we_need - i)
        if _axis_name not in h5file:
            axis_data = h5file.create_dataset(_axis_name, (2,), 'float64')
        else:
            axis_data = h5file[_axis_name]

        # set the extent to the data we have...
        axis_data[0] = data_object.axes[i].min
        axis_data[1] = data_object.axes[i].max

        # fill in any values we have stored in the Axis object
        for key, value in data_object.axes[i].attrs.items():
#             if key == 'UNITS':
#                 try:
#                     axis_data.attrs['UNITS'] = np.array([str(data_object.axes[i].attrs['UNITS']).encode('utf-8')])
#                 except:
#                     axis_data.attrs['UNITS'] = np.array([b'a.u.'])
#             else:
            axis_data.attrs[key] = np.array([value.encode('utf-8')]) if isinstance(value, (str, OSUnits)) else value
    h5file.close()


def write_h5_openpmd(data, filename=None, path=None, dataset_name=None, overwrite=True, axis_name=None,
    time_to_si=1.0, length_to_si=1.0, data_to_si=1.0 ):
    """
    Usage:
        write_h5_openpmd(diag_data, '/path/to/filename.h5')    # writes out Visxd compatible HDF5 data.

    Since h5 format does not support python strings, we will convert all string data (units, names etc)
    to bytes strings before writing.

    see also read_h5() function in this file

    """
    if isinstance(data, H5Data):
        data_object = data
    elif isinstance(data, np.ndarray):
        data_object = H5Data(data)
    else:
        try:  # maybe it's something we can wrap in a numpy array
            data_object = H5Data(np.array(data))
        except:
            raise Exception(
                "Invalid data type.. we need a 'hdf5_data', numpy array, or something that can go in a numy array")

    # now let's make the H5Data() compatible with VisXd and such...
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
        if overwrite:
            os.remove(fname)
        else:
            c = 1
            while os.path.isfile(fname[:-3]+'.copy'+str(c)+'.h5'):
                c += 1
            fname = fname[:-3]+'.copy'+str(c)+'.h5'
    h5file = h5py.File(fname,'a')

    # now put the data in a group called this...
 #   h5dataset = h5file.create_dataset(current_name_attr, data_object.shape, data=data_object.view(np.ndarray))
    # these are required.. so make defaults ones...
 #   h5dataset.attrs['UNITS'], h5dataset.attrs['LONG_NAME'] = np.array([b'']), np.array([b''])
    # convert osunit class back to ascii
    data_attrs = data_object.data_attrs.copy()
    try:
        data_attrs['UNITS'] = np.array([str(data_object.data_attrs['UNITS']).encode('utf-8')])
    except:
        data_attrs['UNITS'] = np.array([b'a.u.'])
    # copy over any values we have in the 'H5Data' object;
#    for key, value in data_attrs.items():
#        h5dataset.attrs[key] = np.array([value.encode('utf-8')]) if isinstance(value, str) else value
    # these are required so we make defaults..
    h5file.attrs['DT'] = [1.0]
    h5file.attrs['ITER'] = [0]
    h5file.attrs['MOVE C'] = [0]
    h5file.attrs['PERIODIC'] = [0]
    h5file.attrs['TIME'] = [0.0]
    h5file.attrs['TIME UNITS'] = [b'1 / \omega_p']
    h5file.attrs['TYPE'] = [b'grid']
    h5file.attrs['XMIN'] = [0.0]
    h5file.attrs['XMAX'] = [0.0]
    h5file.attrs['openPMD'] = np.string_("1.1.0")
    h5file.attrs['openPMDextension'] = np.uint32(0)
    h5file.attrs['iterationEncoding'] = np.string_('fileBased')
    fileroot=str(filename.split('/')[-1])
    fileroot=str(fileroot.split('-')[0])
    h5file.attrs['iterationFormat'] = np.string_("%s-%%T.h5" %fileroot)
    h5file.attrs['basePath']=np.string_('/data/%T/')
    h5file.attrs['meshesPath']=np.string_('mesh/')
    # h5file.attrs['particlesPath']= 'particles/' .encode('utf-8')
    # now make defaults/copy over the attributes in the root of the hdf5

    baseid = h5file.create_group("data")
    iterid = baseid.create_group(str(data.run_attrs['ITER'][0]))


    meshid = iterid.create_group("mesh")
    datasetid = meshid.create_dataset(data_attrs['NAME'], data_object.shape, data=data_object.view(np.ndarray) )

   # h5dataset = datasetid.create_dataset(current_name_attr, data_object.shape, data=data_object.view(np.ndarray))


 #   for key, value in data_object.run_attrs.items():
 #       h5file.attrs[key] = np.array([value.encode('utf-8')]) if isinstance(value, str) else value



    iterid.attrs['dt'] = data.run_attrs['TIME'][0]/float(data.run_attrs['ITER'][0])
    iterid.attrs['time'] = data.run_attrs['TIME'][0] 
    iterid.attrs['timeUnitSI'] = time_to_si


    number_axis_objects_we_need = len(data_object.axes)
 
    # deltax= data.run_attrs['XMAX'] - data.run_attrs['XMIN']

    deltax = np.zeros(number_axis_objects_we_need)
    #  deltax = 
    local_offset = np.arange(number_axis_objects_we_need, dtype = np.float32)
    local_globaloffset = np.arange(number_axis_objects_we_need, dtype = np.float64)
    local_position = np.arange(number_axis_objects_we_need, dtype = np.float32)
    local_position[0] = 0.0
    local_position[1] = 0.0

    local_gridspacing = np.arange(number_axis_objects_we_need, dtype = np.float32)

    if(number_axis_objects_we_need == 1):
        local_axislabels=[b'x']
        deltax[0] = data.axes[0][1]-data.axes[0][0]        
        local_gridspacing=np.float32(deltax)

        local_globaloffset[0] = np.float32(0.0)

        local_offset[0]= np.float32(0.0)

    elif (number_axis_objects_we_need == 2):
        local_axislabels=[b'x', b'y']
        deltax[0] = data.axes[0][1]-data.axes[0][0]        
        deltax[1] = data.axes[1][1]-data.axes[1][0]
        # temp=deltax[0]
        # deltax[0]=deltax[1]
        # deltax[1]=temp
        local_gridspacing=np.float32(deltax)

        local_globaloffset[0] = np.float32(0.0)
        local_globaloffset[1] = np.float32(0.0)

        local_offset[0]= np.float32(0.0)
        local_offset[1]= np.float32(0.0)

    else:
        local_axislabels=[b'x',b'y',b'z']
        deltax[0] = data.axes[0][1]-data.axes[0][0]        
        deltax[1] = data.axes[1][1]-data.axes[1][0]
        deltax[2] = data.axes[2][1]-data.axes[2][0]        
    
        local_gridspacing=np.float32(deltax)

        local_globaloffset[0] = np.float32(0.0)
        local_globaloffset[1] = np.float32(0.0)
        local_globaloffset[2] = np.float32(0.0)

        local_offset[0]= np.float32(0.0)
        local_offset[1]= np.float32(0.0)
        local_offset[2]= np.float32(0.0)


     
    datasetid.attrs['dataOrder'] = np.string_('F')
    datasetid.attrs['geometry'] = np.string_('cartesian')
    datasetid.attrs['geometryParameters'] =  np.string_('cartesian')

    datasetid.attrs['axisLabels'] = local_axislabels
    datasetid.attrs['gridUnitSI'] = np.float64(length_to_si)
    datasetid.attrs['unitSI'] = np.float64(data_to_si)
    datasetid.attrs['position'] = local_position
    datasetid.attrs['gridSpacing'] = local_gridspacing
    datasetid.attrs['gridGlobalOffset'] = local_globaloffset
    datasetid.attrs['time']=data.run_attrs['TIME'][0]
    datasetid.attrs['timeOffset'] = 0.0
    datasetid.attrs['unitDimension'] = ( 0., 1., -2., -1., 0., 0., 0.)
    # datasetid.attrs['dt']=1.0


    # # now go through and set/create our axes HDF entries.
    # if not axis_name:
    #     axis_name = "AXIS/AXIS"
    # for i in range(0, number_axis_objects_we_need):
    #     _axis_name = axis_name + str(number_axis_objects_we_need - i)
    #     if _axis_name not in h5file:
    #         axis_data = h5file.create_dataset(_axis_name, (2,), 'float64')
    #     else:
    #         axis_data = h5file[_axis_name]

    #     # set the extent to the data we have...
    #     axis_data[0] = data_object.axes[i].min
    #     axis_data[1] = data_object.axes[i].max

    #     # fill in any values we have stored in the Axis object
    #     for key, value in data_object.axes[i].attrs.items():
    #         if key == 'UNITS':
    #             try:
    #                 axis_data.attrs['UNITS'] = np.array([str(data_object.axes[i].attrs['UNITS']).encode('utf-8')])
    #             except:
    #                 axis_data.attrs['UNITS'] = np.array([b'a.u.'])
    #         else:
    #             axis_data.attrs[key] = np.array([value.encode('utf-8')]) if isinstance(value, str) else value
    h5file.close()



if __name__ == '__main__':
    import osh5utils as ut
    a = np.arange(6.0).reshape(2, 3)
    ax, ay = DataAxis(0, 3, 3, attrs={'UNITS': '1 / \omega_p'}), DataAxis(10, 11, 2, attrs={'UNITS': 'c / \omega_p'})
    da = {'UNITS': 'n_0', 'NAME': 'test', }
    h5d = H5Data(a, timestamp='123456', data_attrs=da, axes=[ay, ax])
    write_h5(h5d, './test-123456.h5')
    rw = read_h5('./test-123456.h5')
    h5d = read_h5('./test-123456.h5')  # read from file to get all default attrs
    print("rw is h5d: ", rw is h5d, '\n')
    print(repr(rw))

    # let's read/write a few times and see if there are mutations to the data
    # you should also diff the output h5 files
    for i in range(5):
        write_h5(rw, './test' + str(i) + '-123456.h5')
        rw = read_h5('./test' + str(i) + '-123456.h5')
        assert (rw == a).all()
        for axrw, axh5d in zip(rw.axes, h5d.axes):
            assert axrw.attrs == axh5d.attrs
            assert (axrw == axh5d).all()
        assert h5d.timestamp == rw.timestamp
        assert h5d.name == rw.name
        assert h5d.data_attrs == rw.data_attrs
        assert h5d.run_attrs == rw.run_attrs
        print('checking: ', i+1, 'pass completed')

    # test some other functionaries
    print('\n meta info of rw: ', rw)
    print('\nunit of rw is ', rw.data_attrs['UNITS'])
    rw **= 3
    print('unit of rw^3 is ', rw.data_attrs['UNITS'])
    print('contents of rw^3: \n', rw.view(np.ndarray))
