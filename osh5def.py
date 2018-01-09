#!/usr/bin/env python

"""osh5def.py: Define the OSIRIS HDF5 data class and basic functions.
    The basic idea is to make the data unit and axes consistent with the data itself. Therefore users should only modify
    the unit and axes by modifying the data or by dedicated functions (unit conversion for example).
"""

import numpy as np
import re
import copy

# Important: the first occurrence of serial numbers between '-' and '.' must be the time stamp information
fn_rule = re.compile(r'-(\d+).')


class H5Data(np.ndarray):

    def __new__(cls, input_array, timestamp=None, name=None, data_attrs=None, run_attrs=None, axes=None):
        """wrap input_array into our class, and we don't copy the data!"""
        obj = input_array.view(cls)
        if timestamp:
            obj.timestamp = timestamp
        if name:
            obj.name = name
        if data_attrs:
            obj.data_attrs = copy.deepcopy(data_attrs)  # there is OSUnits obj inside
        if run_attrs:
            obj.run_attrs = run_attrs.copy()
        if axes:
            obj.axes = copy.deepcopy(axes)  # the elements are numpy arrays
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.timestamp = getattr(obj, 'timestamp', '0'*6)
        self.name = getattr(obj, 'name', 'data')
        self.data_attrs = getattr(obj, 'data_attrs', {})
        self.run_attrs = getattr(obj, 'run_attrs', {})
        self.axes = getattr(obj, 'axes', [])

    def __str__(self):
        return ''.join([self.name, '-', self.timestamp])

    def __add__(self, other):
        return super(H5Data, self).__add__(other)

    def __sub__(self, other):
        return super(H5Data, self).__sub__(other)

    def __mul__(self, other):
        v = super(H5Data, self).__mul__(other)
        if not isinstance(other, (complex, int, float)):
            v.data_attrs['UNITS'] = self.data_attrs['UNITS'] * other.data_attrs['UNITS']
        return v

    def __truediv__(self, other):
        v = super(H5Data, self).__truediv__(other)
        if not isinstance(other, (complex, int, float)):
            v.data_attrs['UNITS'] = self.data_attrs['UNITS'] / other.data_attrs['UNITS']
        return v

    def __pow__(self, other, modulo=None):
        v = super(H5Data, self).__pow__(other)
        v.data_attrs['UNITS'] = pow(self.data_attrs['UNITS'], other)
        return v

    def __iadd__(self, other):
        return super(H5Data, self).__iadd__(other)

    def __isub__(self, other):
        return super(H5Data, self).__isub__(other)

    def __imul__(self, other):
        if not isinstance(other, (complex, int, float)):
            self.data_attrs['UNITS'] = self.data_attrs['UNITS'] * other.data_attrs['UNITS']
        return self

    def __idiv__(self, other):
        if not isinstance(other, (complex, int, float)):
            self.data_attrs['UNITS'] = self.data_attrs['UNITS'] / other.data_attrs['UNITS']
        return self

    def __ipow__(self, other, modulo=None):
        super(H5Data, self).__ipow__(other)
        self.data_attrs['UNITS'] = pow(self.data_attrs['UNITS'], other)
        return self

    def __getitem__(self, index):
        """I am inclined to support only basic indexing/slicing. Otherwise it is too difficult to define the axes.
             However we would return an ndarray if advace indexing is invoked as it might help things floating...
        """
        v = super(H5Data, self).__getitem__(index)
        # v.axes = copy.deepcopy(self.axes)
        # # let's say a.shape=(4,4), a[1:3] **= 2 won't make sense any way ...
        # v.data_attrs['UNITS'] = copy.deepcopy(self.data_attrs['UNITS'])

        # put everything into a list
        try:
            iter(index)
            idxl = index
        except TypeError:
            idxl = [index]
        try:
            pn, i, stop = 0, 0, len(idxl)
            while i < stop:
                if isinstance(idxl[i], int):  # i is a trivial dimension now
                    del v.axes[i - pn]
                    pn += 1
                elif isinstance(idxl[i], slice):  # also slice the axis
                    v.axes[i] = copy.deepcopy(v.axes[i])  # numpy array deepcopy
                    v.axes[i].ax = v.axes[i].ax[idxl[i]]
                elif idxl[i] is Ellipsis:  # let's jump out and count backward
                    i += self.ndim - stop
                elif idxl[i] is None:
                    pass
                else:  # type not supported
                    return v.view(np.ndarray)
                i += 1
        except AttributeError:  #TODO .axes was lost for some reason, need a better look
            pass
        return v.view(H5Data)

    def meta2dic(self):
        """return a shallow copy of the meta data as a dictionary"""
        return {'timestamp': self.timestamp, 'name': self.name, 'data_attrs': self.data_attrs,
                'run_attrs': self.run_attrs, 'axes': self.axes}

    def transpose(self, *axes):
        v = super(H5Data, self).transpose(*axes)
        if not axes:  # axes is none, numpy default is to reverse the order
            axes = range(len(v.axes)-1, -1, -1)
        v.axes = [self.axes[i] for i in axes]
        return v

    def sum(self, axis=None, out=None, **unused_kw):
        dim = self.ndim
        o = super(H5Data, self).sum(self, axis=axis, out=out)
        if out:
            o = out
        if not axis:  # default is to sum over all axis, return a value
            return o[0]
        # remember axis can be negative
        o.axes = [v for i, v in enumerate(axis) if i not in axis and i+dim not in axis]
        return o

