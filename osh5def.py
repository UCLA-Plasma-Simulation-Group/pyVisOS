#!/usr/bin/env python

"""osh5def.py: Define the OSIRIS HDF5 data class and basic functions."""

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
            obj.data_attrs = copy.deepcopy(data_attrs)
        if run_attrs:
            obj.run_attrs = run_attrs.copy()
        if axes:
            obj.axes = copy.deepcopy(axes)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.timestamp = getattr(obj, 'timestamp', '0'*6)
        self.name = getattr(obj, 'name', 'data')
        self.data_attrs = copy.deepcopy(getattr(obj, 'data_attrs', {}))  # there is OSUnits obj inside
        self.run_attrs = getattr(obj, 'run_attrs', {}).copy()
        self.axes = copy.deepcopy(getattr(obj, 'axes', []))  # the elements are numpy arrays

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
                    v.axes.pop(i - pn)
                    pn += 1
                elif isinstance(idxl[i], slice):  # also slice the axis
                    v.axes[i].axisdata = v.axes[i].axisdata[idxl[i]]
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

