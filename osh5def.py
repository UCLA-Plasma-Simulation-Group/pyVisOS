#!/usr/bin/env python

"""osh5def.py: Define the OSIRIS HDF5 data class and basic functions.
    The basic idea is to make the data unit and axes consistent with the data itself. Therefore users should only modify
    the unit and axes by modifying the data or by dedicated functions (unit conversion for example).
"""

import numpy as np
import re
import copy
from fractions import Fraction as frac


class DataAxis:
    def __init__(self, axis_min, axis_max, axis_npoints, attrs=None):
        # attrs should be a dictionary
        if axis_min > axis_max:
            raise Exception('illegal axis range: [ %(l)s, %(r)s ]' % {'l': axis_min, 'r': axis_max})
        self.ax = np.arange(axis_min, axis_max, (axis_max - axis_min) / axis_npoints)
        # now make attributes for axis that are required..
        self.attrs = {'UNITS': OSUnits('a.u.'), 'LONG_NAME': "", 'NAME': ""}
        # get the attributes for the AXIS
        if attrs:
            self.attrs.update(attrs)

    def __str__(self):
        return ''.join([str(self.attrs['NAME']), ': [', str(self.ax[0]), ', ', str(self.ax[-1]), '] ',
                        str(self.attrs['UNITS'])])

    def __repr__(self):
        if len(self.ax) == 0:
            return 'None'
        return ''.join([str(self.__class__.__module__), '.', str(self.__class__.__name__), ' at ', hex(id(self)),
                        ': size=', str(self.ax.size), ', (min, max)=(', repr(self.ax[0]), ', ',
                        repr(self.max), '), ', repr(self.attrs)])

    def __getitem__(self, index):
        return self.ax[index]

    def __eq__(self, other):
        return (self.ax == other.ax).all()

    # def __getstate__(self):
    #     return self.ax[0], self.ax[-1], self.size, self.attrs
    #
    # def __setstate__(self, state):
    #     self.ax = np.linspace(state[0], state[1], state[2])
    #     self.attrs = state[3]

    @property
    def min(self):
        return self.ax[0]

    @property
    def max(self):
        try:
            return self.ax[-1] + self.ax[1] - self.ax[0]
        except IndexError:
            return self.ax[-1]

    @property
    def size(self):
        return self.ax.size

    def __len__(self):
        return self.ax.size

    @property
    def increment(self):
        try:
            return self.ax[1] - self.ax[0]
        except IndexError:
            return 0

    def to_phys_unit(self, wavelength=None, density=None):
        """
        convert this axis to physical units. note that this function won't change the actual axis.
        the copy of the axis data is returned
        :param wavelength: laser wavelength in micron
        :param density: critical plasma density in cm^-3
        :return: a converted axes, unit
        """
        if not wavelength:
            if not density:
                wavelength = 0.351
                density = 1.12e21
            else:
                wavelength = 1.98e10 * np.sqrt(1./density)
        elif not density:
            density = 3.93e20 * wavelength**2
        if self.attrs['UNITS'].is_frequency():
            return self.ax * 2.998e2 / wavelength, 'THz'
        if self.attrs['UNITS'].is_time():
            return self.ax * wavelength * 5.31e-4, 'ps'
        if self.attrs['UNITS'].is_length():
            return self.ax * wavelength / (2 * np.pi), '\mu m'
        if self.attrs['UNITS'].is_density():
            return self.ax * density, 'cm^{-3}'


class OSUnits:
    name = ['m_e', 'c', '\omega_p', 'e', 'n_0']
    xtrnum = re.compile(r"(?<=\^)\d+|(?<=\^{).*?(?=})")

    def __init__(self, s):
        """
        :param s: string notation of the units. there should be whitespace around quantities and '/' dividing quantities
        """
        self.power = np.array([frac(0), frac(0), frac(0), frac(0), frac(0)])
        # if isinstance(s, bytes):
        #     s = s.decode("utf-8")
        if 'a.u.' != s:
            sl = s.split()
            nominator = True
            while sl:
                ss = sl.pop(0)
                if ss == '/':
                    nominator = False
                    continue
                for p, n in enumerate(OSUnits.name):
                    if n == ss[0:len(n)]:
                        res = OSUnits.xtrnum.findall(ss)  # extract numbers
                        if res:
                            self.power[p] = frac(res[0]) if nominator else -frac(res[0])
                        else:
                            self.power[p] = frac(1, 1) if nominator else frac(-1, 1)
                        break
                    elif ss in ['1', '2', '\pi', '2\pi']:
                        break
                else:
                    raise KeyError('Unknown unit: ' + re.findall(r'\w+', ss)[0])

    def tex(self):
        return '$' + self.__str__() + '$'

    def limit_denominator(self, max_denominator=64):
        """call fractions.Fraction.limit_denominator method for each base unit"""
        self.power = np.array([u.limit_denominator(max_denominator=max_denominator) for u in self.power])

    def is_time(self):
        return (self.power == np.array([frac(0), frac(0), frac(-1), frac(0), frac(0)])).all()

    def is_frequency(self):
        return (self.power == np.array([frac(0), frac(0), frac(1), frac(0), frac(0)])).all()

    def is_velocity(self):
        return (self.power == np.array([frac(0), frac(1), frac(0), frac(0), frac(0)])).all()

    def is_length(self):
        return (self.power == np.array([frac(0), frac(1), frac(-1), frac(0), frac(0)])).all()

    def is_density(self):
        return (self.power == np.array([frac(0), frac(0), frac(0), frac(0), frac(1)])).all()

    def __mul__(self, other):
        res = OSUnits('')
        res.power = self.power + other.power
        return res

    def __truediv__(self, other):
        res = OSUnits('')
        res.power = self.power - other.power
        return res

    def __pow__(self, other, modulo=1):
        res = OSUnits('')
        res.power = self.power * frac(other)
        return res

    def __eq__(self, other):
        return (self.power == other.power).all()

    def __str__(self):
        disp = ''.join(['' if p == 0 else n + " " if p == 1 else ''.join([n, '^{', str(p), '} '])
                        for n, p in zip(OSUnits.name, self.power)])
        if not disp:
            return 'a.u.'
        return disp

    def __repr__(self):
        return ''.join([str(self.__class__.__module__), '.', str(self.__class__.__name__), ' at ', hex(id(self)),
                        ': ', repr(self.name), '=[', ', '.join([str(fr) for fr in self.power]), ']'])


# Important: the first occurrence of serial numbers between '-' and '.' must be the time stamp information
fn_rule = re.compile(r'-(\d+)\.')


class H5Data(np.ndarray):

    def __new__(cls, input_array, timestamp=None, name=None, data_attrs=None, run_attrs=None, axes=None):
        """wrap input_array into our class, and we don't copy the data!"""
        obj = np.asarray(input_array).view(cls)
        if timestamp:
            obj.timestamp = timestamp
        if name:
            obj.name = name
        if data_attrs:
            obj.data_attrs = copy.deepcopy(data_attrs)  # there is OSUnits obj inside
        if run_attrs:
            obj.run_attrs = run_attrs
        if axes:
            obj.axes = copy.deepcopy(axes)   # the elements are numpy arrays
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.timestamp = getattr(obj, 'timestamp', '0' * 6)
        self.name = getattr(obj, 'name', 'data')
        if self.base is obj:
            self.data_attrs = getattr(obj, 'data_attrs', {})
            self.run_attrs = getattr(obj, 'run_attrs', {})
            self.axes = getattr(obj, 'axes', [])
        else:
            self.data_attrs = copy.deepcopy(getattr(obj, 'data_attrs', {}))
            self.run_attrs = copy.deepcopy(getattr(obj, 'run_attrs', {}))
            self.axes = copy.deepcopy(getattr(obj, 'axes', []))

    @property
    def T(self):
        return self.transpose()

    # need the following two function for mpi4py high level function to work correctly
    def __setstate__(self, state, *args):
        self.__dict__ = state[-1]
        super(H5Data, self).__setstate__(state[:-1], *args)

    # It looks like mpi4py/ndarray use reduce for pickling. One would think setstate/getstate pair should also work but
    # it turns out the __getstate__() function is never called!
    # Luckily ndarray doesn't use __dict__ so we can pack everything in it.
    def __reduce__(self):
        ps = super(H5Data, self).__reduce__()
        ms = ps[2] + (self.__dict__,)
        return ps[0], ps[1], ms

    def __getstate__(self):
        return self.__reduce__()

    def __str__(self):
        return ''.join([self.name, '-', self.timestamp, ' of shape ', str(self.shape)])

    def __repr__(self):
        return ''.join([str(self.__class__.__module__), '.', str(self.__class__.__name__), ' at ', hex(id(self)),
                        ', shape', str(self.shape), ',\naxis:\n  ',
                        '\n  '.join([repr(ax) for ax in self.axes]) if len(self.axes) else 'None',
                        '\ndata_attrs: ', repr(self.data_attrs), '\nrun_attrs:', repr(self.run_attrs)])

    def __getitem__(self, index):
        """I am inclined to support only basic indexing/slicing. Otherwise it is too difficult to define the axes.
             However we would return an ndarray if advance indexing is invoked as it might help things floating...
        """
        ndim = self.ndim
        try:
            v = super(H5Data, self).__getitem__(index)
        except IndexError:
            return self.view(np.ndarray)  # maybe it is a scalar
        try:
            iter(index)
            idxl = index
        except TypeError:
            idxl = [index]
        stop = len(idxl)
        dn = 0
        for i in range(len(idxl)):
            if isinstance(idxl[i], int):  # i is a trivial dimension now
                del v.axes[i - dn]
                dn += 1
            elif isinstance(idxl[i], slice):  # also slice the axis
                v.axes[i - dn].ax = v.axes[i - dn].ax[idxl[i]]
            elif idxl[i] is Ellipsis:  # let's fast forward to the next explicitly referred axis
                # i += ndim - stop
                dn -= ndim - stop
            elif idxl[i] is None:  # in numpy None means newAxis
                v.axes.insert(i - dn, DataAxis(0., 1., 1))
            else:  # type not supported
                return v.view(np.ndarray)
        return v

    def meta2dict(self):
        """return a deep copy of the meta data as a dictionary"""
        return copy.deepcopy(self.__dict__)

    def transpose(self, *axes):
        v = super(H5Data, self).transpose(*axes)
        if axes is () or axes[0] is None:  # axes is none, numpy default is to reverse the order
            axes = range(len(v.axes)-1, -1, -1)
        v.axes = [self.axes[i] for i in axes]
        return v

    def __del_axis(self, axis):
        # axis cannot be None
        if isinstance(axis, int):
            del self.axes[axis]
        else:
            # remember axis index can be negative
            self.axes = [v for i, v in enumerate(self.axes) if i not in axis and i - self.ndim not in axis]

    # handel extra metadata when calling reduce methods like sum, min/max etc
    def __handle_reduce_ex(self, out=None, axis=None, keepdims=False):
        if not keepdims:
            if axis is None:  # default is to reduce over all axis, return a value
                axis = range(0, self.ndim)
            self.__del_axis(axis)
            if out:
                out.__del_axis(axis)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        o = super(H5Data, self).mean(axis=axis, out=out, dtype=dtype, keepdims=keepdims)
        o.__handle_reduce_ex(out=out, axis=axis, keepdims=keepdims)
        return o

    def sum(self, axis=None, out=None, dtype=None, keepdims=False):
        o = super(H5Data, self).sum(axis=axis, out=out, dtype=dtype, keepdims=keepdims)
        o.__handle_reduce_ex(out=out, axis=axis, keepdims=keepdims)
        return o

    def min(self, axis=None, out=None, keepdims=False):
        o = super(H5Data, self).min(axis=axis, out=out, keepdims=keepdims)
        o.__handle_reduce_ex(out=out, axis=axis, keepdims=keepdims)
        return o

    def max(self, axis=None, out=None, keepdims=False):
        o = super(H5Data, self).max(axis=axis, out=out, keepdims=keepdims)
        o.__handle_reduce_ex(out=out, axis=axis, keepdims=keepdims)
        return o

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        o = super(H5Data, self).std(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
        o.__handle_reduce_ex(out=out, axis=axis, keepdims=keepdims)
        return o

    def argmax(self, axis=None, out=None):
        o = super(H5Data, self).argmax(axis=axis, out=out)
        o.__handle_reduce_ex(out=out, axis=axis, keepdims=False)
        return

    def argmin(self, axis=None, out=None):
        o = super(H5Data, self).argmin(axis=axis, out=out)
        o.__handle_reduce_ex(out=out, axis=axis, keepdims=False)
        return o

    def ptp(self, axis=None, out=None):
        o = super(H5Data, self).ptp(axis=axis, out=out)
        o.__handle_reduce_ex(out=out, axis=axis, keepdims=False)
        return o

    def swapaxes(self, axis1, axis2):
        o = super(H5Data, self).swapaxes(axis1, axis2)
        o.axes[axis1], o.axes[axis2] = o.axes[axis2], o.axes[axis1]
        return o

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        o = super(H5Data, self).var(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
        o.__handle_reduce_ex(out=out, axis=axis, keepdims=False)
        return o

    def squeeze(self, axis=None):
        v = super(H5Data, self).squeeze(axis=axis)
        if axis is None:
            axis = [i for i, d in enumerate(self.shape) if d == 1]
        for i in reversed(axis):
            del v.axes[i]
        return v

    def __array_wrap__(self, out, context=None):
        """Here we handle the unit attribute
        We do not check the legitimacy of ufunc operating on certain unit. We hard code a few unit changing
        rules according to what ufunc is called
        """
        # the document says that __array_wrap__ could be deprecated in the future but this is the most direct way...
        div, mul = 1, 2
        op, __ufunc_mapping = None, {'sqrt': '1/2', 'cbrt': '1/3', 'square': '2', 'power': '',
                                     'true_divide': div, 'floor_divide': div, 'reciprocal': '-1', 'multiply': mul}
        if context:
            op = __ufunc_mapping.get(context[0].__name__)
        if op is not None:
            try:
                if isinstance(op, str):
                    if not op:  # op is 'power', get the second operand
                        op = context[1][1]
                    out.data_attrs['UNITS'] **= op
                elif op == div:  # op is divide
                    if not isinstance(context[1][0], H5Data):  # nominator has no unit
                        out.data_attrs['UNITS'] **= '-1'
                    else:
                        out.data_attrs['UNITS'] = context[1][0].data_attrs['UNITS'] / context[1][1].data_attrs['UNITS']
                else:  # op is multiply
                    out.data_attrs['UNITS'] = context[1][0].data_attrs['UNITS'] * context[1][1].data_attrs['UNITS']
            except (AttributeError, KeyError):  # .data_attrs['UNITS'] failed
                pass
        return np.ndarray.__array_wrap__(self, out, context)

    @staticmethod
    def __get_axes_bound(axes, bound):  # bound should have depth of 2

        def get_index(ax, bd):
            tmp = [int(round((co - ax.min) / ax.increment)) if co else None for co in bd]
            # clip to legal range
            if not tmp[0] or tmp[0] < 0:
                tmp[0] = 0
            if not tmp[1] or tmp[1] > ax.size:
                tmp[1] = ax.size
            return tmp

        # i keeps track of the axes we are working on
        ind, i = [], 0
        for bnd in bound:
            if bnd is Ellipsis:
                # fill in all missing axes
                for j in range(len(axes) - len(bound) + 1):
                    ind.append(slice(None))
                    i += 1
                continue
            elif not bnd:  # None or empty list/tuple or 0
                ind.append(slice(None))
            elif len(bnd) == 3:  # slice with step, None can appear at any places
                sgn = int(np.sign(bnd[2]))
                if sgn == 1:
                    se = get_index(axes[i], bnd[0:2])
                else:
                    se = get_index(axes[i], reversed(bnd[0:2]))
                    se = list(reversed(se))
                step = int(round(bnd[2] / axes[i].increment))
                se.append(step if abs(step) > 0 else sgn)
                ind.append(slice(*se))
            else:  # len(bnd) == 1 or 2
                ind.append(slice(*get_index(axes[i], bnd)))
            i += 1
        return ind

    # get the depth of list/tuple recursively, empty list/tuple will raise ValueError
    @staticmethod
    def __check_bound_depth(bnd):
        return max(H5Data.__check_bound_depth(b) for b in bnd) + 1 if isinstance(bnd, (tuple, list, np.ndarray)) else 0

    @staticmethod
    def __get_symmetric_bound(axes, index):
        return [slice(*[None if idx.stop is None else ax.size - idx.stop,
                        None if idx.start is None else ax.size - idx.start,
                        idx.step]) for ax, idx in zip(axes, index)]

    def subrange(self, bound=None, new=False):
        """
        use .axes[:] data to index the array
        :param bound: see bound as using slice indexing with floating points. [1:3, :, 2:9:2, ..., :8] for a 6-D array
            with dx=0.1 in all directions would look like bound=[(0.1, 0.3), None, (0.2, 0.9, 0.2), ..., (None, 0.8)]
        :param new: if true return a new array instead of a view
        """
        if not bound:  # None or empty list/tuple or 0
            return self
        if H5Data.__check_bound_depth(bound) == 1:
            bound = (bound,)
        index = self.__get_axes_bound(self.axes, bound)
        if new:
            return copy.deepcopy(self[index])
        else:
            return self[index]

    def set_value(self, bound=None, val=(0,), symmetric=True, inverse_select=False, method=None):
        """
        set values inside bound using val
        :param bound: lists of triples of elements marking the lower and higher bound of the bound, e.g.
                [(l1,u1), (l2,u2), (l3,u3)] in 3d data marks data[l1<z<u1, l2<y<u2, l3<x<u3]. See bound parameter
                in function subrange for more detail. There can be multiple lists like this marking multiple regions.
                Note that if ln or un is out of bound they will be clipped to the array boundary.
        :param val: what value to set, default is (0,). val can also be list of arrays with the same size
                as each respective bound.
        :param inverse_select: set array elements OUTSIDE specified bound to val. This will used more memory
                as temp array will be created. If true val can only be a number.
        :param symmetric: if true then the mask will also apply to the center-symmetric regions where center is define
                as nx/2 for each dimension
        :param method: how to set the value. if None the val will be assigned. User can provide function that accept
                two parameters, and the result is self[bound[i]] = user_func(self[bound[i]], val[i]). Note that
                method always acts on the specified bound regardless of inverse_select being true or not.
        """
        if not bound:
            return
        # convert to ndarray for maximum compatibility
        v = self.view(np.ndarray)
        bdp = H5Data.__check_bound_depth(bound)
        if bdp == 1:  # probably 1D data
            bound = ((bound,),)
        elif bdp == 2:  # only one region is set
            bound = (bound,)
        elif bdp > 3:  # 3 level max: list of regions; list of bounds in a region; start, end, step in one bound
            raise ValueError('Too many levels in the bound parameter')
        try:
            iter(val)
        except TypeError:
            val = (val,)
        rec, vallen = [], len(val)
        for i, bnd in enumerate(bound):
            index = [self.__get_axes_bound(self.axes, bnd)]
            if symmetric:
                index.append(H5Data.__get_symmetric_bound(self.axes, index[0]))
            for idx in index:
                if inverse_select:  # record original data for later use
                    if method:
                        rec.append((idx, copy.deepcopy(method(v[idx], val[i % vallen]))))
                    else:
                        rec.append((idx, copy.deepcopy(v[idx])))
                else:
                    if method:
                        v[idx] = method(v[idx], val[i % vallen])
                    else:
                        v[idx] = val[i % vallen]
        if inverse_select:
            # set everything to val and restore the original data in regions
            self.fill(val[0])
            for r in rec:
                v[r[0]] = r[1]

