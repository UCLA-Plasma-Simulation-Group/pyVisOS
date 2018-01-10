"""Provide basic operations for H5Data"""

import osh5def
import osaxis
import numpy as np
import copy


def stack(arr, axis=0, axesdata=None):
    try:
        if not isinstance(arr[-1], osh5def.H5Data):
            raise TypeError('Input array must contain H5Data objects')
    except (TypeError, IndexError):   # not an array or an empty array, just return what ever passed in
        return arr
    md = arr[-1]
    ax = copy.deepcopy(md.axes)
    if axesdata:
        if axesdata.size() != len(arr):
            raise ValueError('Number of points in axesdata is different from the new dimension to be created')
        ax.insert(axis, axesdata)
    else:  # we assume the new dimension is time
        taxis_attrs = {'UNITS': "1 / \omega_p", 'LONG_NAME': "time", 'NAME': "t"}
        ax.insert(axis, osaxis.DataAxis(arr[0].run_attrs['TIME'],
                                        arr[-1].run_attrs['TIME'], np.size(arr), attrs=taxis_attrs))
    r = np.stack(arr, axis=axis)
    return osh5def.H5Data(r, md.timestamp, md.name, md.data_attrs, md.run_attrs, axes=ax)

