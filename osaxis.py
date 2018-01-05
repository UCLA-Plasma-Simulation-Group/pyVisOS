#!/usr/bin/env python

"""osaxis.py: Define the axis class for OSIRIS output."""

import numpy as np
import copy


class DataAxis:
    def __init__(self, axis_min, axis_max, axis_npoints, attrs=None):
        # self.axis_min = axis_min
        # self.axis_max = axis_max
        # self.axis_nx = axis_npoints
        # self.increment = (self.axis_max - self.axis_min) / self.axis_nx
        self.axisdata = np.arange(axis_min, axis_max, (axis_max - axis_min)/axis_npoints)
        self.attrs = {}
        # get the attributes for the AXIS
        for key, value in attrs.items():
            self.attrs[key] = value

    def clone(self):
        out = copy.deepcopy(self)
        return out

    # def get_axis_points(self):
    #     return np.arange(self.axis_min, self.axis_max, self.increment)

    def axis_min(self):
        return self.axisdata[0]

    def axis_max(self):
        return self.axisdata[-1]

    def axis_npoints(self):
        return self.axisdata.size()

    def increment(self):
        return (self.axisdata[-1] - self.axisdata[0]) / self.axisdata.size()

    # def trim(self, n_pts_start, n_pts_end):
    #     self.axis_nx -= (n_pts_start + n_pts_end)
    #     if self.axis_nx < 0:
    #         raise ValueError('ERROR: empty axis!')
    #     self.axis_min += n_pts_start * self.increment
    #     self.axis_max -= n_pts_end * self.increment
