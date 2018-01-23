#!/usr/bin/env python

"""osaxis.py: Define the axis class for OSIRIS output."""

import numpy as np


class DataAxis:
    def __init__(self, axis_min, axis_max, axis_npoints, attrs=None):
        # attrs should be a dictionary
        if axis_min >= axis_max:
            raise Exception('illegal axis range: [ %(l)s, %(r)s ]' % {'l': axis_min, 'r': axis_max})
        self.ax = np.arange(axis_min, axis_max, (axis_max - axis_min) / axis_npoints)
        # now make attributes for axis that are required..
        self.attrs = {'UNITS': "", 'LONG_NAME': "", 'NAME': ""}
        # get the attributes for the AXIS
        if attrs:
            self.attrs.update(attrs)

    def __str__(self):
        return ''.join([str(self.attrs['NAME']), ': [', str(self.ax[0]), ', ', str(self.ax[-1]), '] ',
                        self.attrs['UNITS']])

    def __repr__(self):
        return ''.join([str(self.__class__.__module__), '.', str(self.__class__.__name__), ' at ', hex(id(self)),
                        ': size=', str(self.ax.size), ', (min, max)=(', repr(self.ax[0]), ', ',
                        repr(self.max()), '), ', repr(self.attrs)])

    def __getitem__(self, index):
        return self.ax[index]

    def __eq__(self, other):
        return (self.ax == other.ax).all()

    # def __getstate__(self):
    #     return self.ax[0], self.ax[-1], self.size(), self.attrs
    #
    # def __setstate__(self, state):
    #     self.ax = np.linspace(state[0], state[1], state[2])
    #     self.attrs = state[3]

    def min(self):
        return self.ax[0]

    def max(self):
        try:
            return self.ax[-1] + self.ax[1] - self.ax[0]
        except IndexError:
            return self.ax[-1]

    def size(self):
        return self.ax.size

    def increment(self):
        try:
            return self.ax[1] - self.ax[0]
        except IndexError:
            return 0


if __name__ == '__main__':
    a = DataAxis(0,10,11, attrs={'UNITS':'c'})
    print(type(a.size()))
    print(a)
    print(repr(a))
    # a[1:3] = [9, 9]  # we don't allow this
