#!/usr/bin/env python

"""osaxis.py: Define the axis class for OSIRIS output."""

import numpy as np


class DataAxis:
    def __init__(self, axis_min, axis_max, axis_npoints, attrs=None):
        self.axisdata = np.linspace(axis_min, axis_max, axis_npoints)
        # now make attributes for axis that are required..
        self.attrs = {'UNITS': b"", 'LONG_NAME': b"", 'TYPE': b"", 'NAME': b""}
        # get the attributes for the AXIS
        try:
            for key, value in attrs.items():
                self.attrs[key] = value
        except AttributeError:  # not a big deal if we can't read attrs (?)
            pass

    def __str__(self):
        return str(self.attrs['NAME']) + ' axis'

    def axis_min(self):
        return self.axisdata[0]

    def axis_max(self):
        return self.axisdata[-1]

    def axis_npoints(self):
        return self.axisdata.size()

    def increment(self):
        try:
            return self.axisdata[1] - self.axisdata[0]
        except IndexError:
            return 0

