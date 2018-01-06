#!/usr/bin/env python

"""osh5def.py: Define the OSIRIS HDF5 data class and basic functions."""


from osaxis import *
import re


class HDFData:
    """Important: the first occurrence of serial numbers between '-' and '.' must be the time stamp information"""
    fn_rule = re.compile(r'-(\d+).')

    def __init__(self):
        self.timestamp = '0'*6   # default use six digits for time stamp
        self.name = 'data'       # default name of the dataset, also use as prefix when write_hdf
        self.sim_dim = None      # xd simulation
        self.data_attrs = {}     # attributes related to the data array
        self.run_attrs = {}      # attributes related to the run
        self.axes = []           # data axis
        self.data = None         # this is what holding the actual dataset

    def __str__(self):
        return self.name

    def clone(self, meta_data_only=False):
        if meta_data_only:
            n = HDFData()
            n.timestamp, n.name, n.sim_dim, n.data_attrs, n.run_attrs, n.axes = \
                self.timestamp, self.name, self.sim_dim, \
                copy.deepcopy(self.data_attrs), copy.copy(self.run_attrs), copy.copy(self.axes)
            return n
        else:
            return copy.deepcopy(self)

    def __getitem__(self, item):
        v = self.clone(meta_data_only=True)
        v.data = self.data[item]
        for i, j in enumerate(item):
            v.axes[i].axisdata = v.axes[i].axisdata[j]
        return v

    def __add__(self, other):
        if self.data_attrs['UNITS'][0] != other.data_attrs['UNITS'][0]:
            raise TypeError('Error: adding quantities with different units')
        v = self.clone(meta_data_only=True)
        v.data = self.data + other.data
        return v

    def __sub__(self, other):
        if self.data_attrs['UNITS'][0] != other.data_attrs['UNITS'][0]:
            raise TypeError('Error: subtracting quantities with different units')
        v = self.clone(meta_data_only=True)
        v.data = self.data - other.data
        return v

    def __mul__(self, other):
        v = self.clone(meta_data_only=True)
        if not isinstance(other, (complex, int, float)):
            v.data_attrs['UNITS'][0] = self.data_attrs['UNITS'][0] * other.data_attrs['UNITS'][0]
        v.data = self.data * other.data
        return v

    def __truediv__(self, other):
        v = self.clone(meta_data_only=True)
        if not isinstance(other, (complex, int, float)):
            v.data_attrs['UNITS'][0] = self.data_attrs['UNITS'][0] / other.data_attrs['UNITS'][0]
        v.data = self.data / other.data
        return v

    def __pow__(self, other, modulo=None):
        v = self.clone(meta_data_only=True)
        v.data_attrs['UNITS'][0] = pow(self.data_attrs['UNITS'][0], other)
        v.data = np.power(self.data, other)
        return v

    def get_axis(self, axis_index):
        return self.axes[axis_index]

    def __remove_axis(self, axis_index):
        del self.axes[axis_index]

    def remove_axis(self, axis_index):
        self.__remove_axis(axis_index)

    def __axis_exists(self, axis_index):
        for (i, axis) in enumerate(self.axes):
            if axis.axis_number == axis_index:
                return True
        return False
