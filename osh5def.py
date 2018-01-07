#!/usr/bin/env python

"""osh5def.py: Define the OSIRIS HDF5 data class and basic functions."""

from osaxis import *
import numpy as np
import re

# Important: the first occurrence of serial numbers between '-' and '.' must be the time stamp information
fn_rule = re.compile(r'-(\d+).')


class H5Data(np.ndarray):

    def __new__(cls, input_array, timestamp=None, name=None, data_attrs=None, run_attrs=None, axes=None):
        obj = input_array.view(cls)
        if timestamp:
            obj.timestamp = timestamp
        if name:
            obj.name = name
        if data_attrs:
            obj.data_attrs = data_attrs.copy()
        if run_attrs:
            obj.run_attrs = run_attrs.copy()
        if axes:
            obj.axes = axes.copy()
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.timestamp = getattr(obj, 'timestamp', '0'*6)
        self.name = getattr(obj, 'name', 'data')
        self.data_attrs = getattr(obj, 'data_attrs', {}).copy()
        self.run_attrs = getattr(obj, 'run_attrs', {}).copy()
        self.axes = getattr(obj, 'axes', []).copy()

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
        self = super(H5Data, self).__ipow__(other)
        self.data_attrs['UNITS'] = pow(self.data_attrs['UNITS'], other)
        return self

