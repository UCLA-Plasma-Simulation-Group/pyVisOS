"""Provide basic operations for H5Data"""

import osh5def
import numpy as np
import copy
import re
from functools import wraps, partial


def metasl(func, mapping=(0, 0)):
    """save meta data before calling the function and restore them to output afterwards
    The input to output mapping is specified in the keyword "mapping" where (i, o) is the
    position of i-th input parameters for which meta data will be saved and the o-th output
    return values that the meta data will be restored to. Multiple save/restore should be
    written as ((i1,o1), (i2,o2), (i3,o3)) etc. The counting start from 0
    """
    @wraps(func)
    def sl(*args, **kwargs):
        # search for all specified input arguments
        saved, kl = [], []
        if isinstance(mapping[0], tuple):  # multiple save/load
            kl = sorted(list(mapping), key=lambda a: a[0])  # sort by input param. pos.
        else:
            kl = [mapping]
        if len(args) < kl[-1][0]:  # TODO(1) they can be in the kwargs
            raise Exception('Cannot find the ' + str(kl[-1][0]) + '-th argument for meta data saving')
        # save meta data into a list
        for tp in kl:
            if hasattr(args[tp[0]], 'meta2dict'):
                saved.insert(0, (args[tp[0]].meta2dict(), tp[1]))
        if not iter(saved):
            raise ValueError('Illegal mapping parameters')
        # execute user function
        out = func(*args, **kwargs)
        # load saved meta data into specified positions
        ol, tl = [], out if isinstance(out, tuple) else (out, )
        try:
            for tp in saved:
                if isinstance(tl[tp[1]], osh5def.H5Data):
                    ol.append(osh5def.H5Data.__dict__.update(tp[0]))
                else:
                    aaa = tl[tp[1]]
                    ol.append(osh5def.H5Data(aaa, **tp[0]))
        except IndexError:
            raise IndexError('Output does not have ' + str(tp[1]) + ' elements')
        except:
            raise TypeError('Output[' + str(tp[1]) + '] is not/cannot convert to H5Data')
        tmp = tuple(ol) if len(ol) > 1 else ol[0] if ol else out
        return tmp
    return sl


def stack(arr, axis=0, axesdata=None):
    """Similar to numpy.stack. Arr is the array of H5Data to be stacked. By default the newly created dimension
    will be labeled as time axis. Other meta data will be copied from the last element of arr
    """
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
        ax.insert(axis, osh5def.DataAxis(arr[0].run_attrs['TIME'],
                                        arr[-1].run_attrs['TIME'], len(arr), attrs=taxis_attrs))
    r = np.stack(arr, axis=axis)
    return osh5def.H5Data(r, md.timestamp, md.name, md.data_attrs, md.run_attrs, axes=ax)


# #----------------------------------- FFT Wrappers ----------------------------------------
# sfunc: for shifting; ffunc: for calculating frequency; ftfunc: for fft the data; uafunc: for updating axes
#
def __idle(a, *args, **kwargs):
    return a


def __try_update_axes(updfunc):
    def update_axes(a, idx, shape, sfunc=__idle, ffunc=__idle):
        if not hasattr(a, 'axes'):  # no axes found
            return
        try:
            iter(idx)
        except TypeError:
            idx = tuple(range(len(a.axes))) if idx is None else (idx,)
        try:
            iter(shape)
        except TypeError:
            shape = (shape,)
        # The data size can change due to the s (or n) keyword. We have to force axes update somehow.
        updfunc(a.axes, idx, shape, sfunc=sfunc, ffunc=ffunc)
    return update_axes


@__try_update_axes
def _update_fft_axes(axes, idx, shape, sfunc, ffunc):
    key, en = ['NAME', 'LONG_NAME', 'UNITS'], ['K(', 'K(', '1/(']
    for i in idx:
        axes[i].attrs.setdefault('shift', axes[i].min())  # save lower bound. value of axes
        axes[i].ax = sfunc(ffunc(shape[i], d=axes[i].increment())) * 2 * np.pi
        for k, e in zip(key, en):
            try:
                axes[i].attrs[k] = ''.join([e, axes[i].attrs[k], ')'])
            except (KeyError, AttributeError):
                pass


@__try_update_axes
def _update_ifft_axes(axes, idx,  shape, sfunc, ffunc):
    key, en = ['NAME', 'LONG_NAME', 'UNITS'], [2, 2, 3]
    for i in idx:
        axes[i].ax = ffunc(shape[i], d=axes[i].increment(), min=axes[i].attrs.get('shift', 0))
        for k, e in zip(key, en):
            try:
                axes[i].attrs[k] = axes[i].attrs[k][e:-1]
            except (KeyError, AttributeError):
                pass


def _get_ihfft_axis(n, d=1.0, min=0.0):
    length = 2 * np.pi / d
    return np.arange(min, length + min, length / n)[0: n//2+1]


def _get_ifft_axis(n, d=1.0, min=0.0):
    length = 2 * np.pi / d
    return np.arange(min, length + min, length / n)


def __ft_interface(ftfunc, sfunc):
    @metasl
    def ft_interface(a, s, axes, norm):
        # call fft and shift the result
        return sfunc(ftfunc(sfunc(a, axes), s, axes, norm), axes)
    return ft_interface


def __shifted_ft_gen(ftfunc, sfunc, ffunc, uafunc):
    def shifted_fft(a, s=None, axes=None, norm=None):
        shape = s if s is not None else a.shape
        o = __ft_interface(ftfunc, sfunc=sfunc)(a, s=s, axes=axes, norm=norm)
        uafunc(o, axes, shape, sfunc=sfunc, ffunc=ffunc)
        return o
    return shifted_fft


# # ========  Normal FFT  ==========
__shifted_fft = partial(__shifted_ft_gen, sfunc=np.fft.fftshift, ffunc=np.fft.fftfreq, uafunc=_update_fft_axes)
__shifted_ifft = partial(__shifted_ft_gen, sfunc=np.fft.ifftshift, ffunc=_get_ifft_axis, uafunc=_update_ifft_axes)


def fftn(a, s=None, axes=None, norm=None):
    return __shifted_fft(np.fft.fftn)(a, s=s, axes=axes, norm=norm)


def fft2(a, s=None, axes=(-2, -1), norm=None):
    return __shifted_fft(np.fft.fft2)(a, s=s, axes=axes, norm=norm)


def fft(a, n=None, axis=-1, norm=None):
    return __shifted_fft(np.fft.fft)(a, s=n, axes=axis, norm=norm)


def ifftn(a, s=None, axes=None, norm=None):
    return __shifted_ifft(np.fft.ifftn)(a, s=s, axes=axes, norm=norm)


def ifft2(a, s=None, axes=(-2, -1), norm=None):
    return __shifted_ifft(np.fft.ifft2)(a, s=s, axes=axes, norm=norm)


def ifft(a, n=None, axis=-1, norm=None):
    # if axes is None:
    #     axes = -1
    return __shifted_ifft(np.fft.ifft)(a, s=n, axes=axis, norm=norm)


# # ========  real FFT  ==========
__shifted_rfft = partial(__shifted_ft_gen, sfunc=__idle, ffunc=np.fft.rfftfreq, uafunc=_update_fft_axes)
__shifted_irfft = partial(__shifted_ft_gen, sfunc=__idle, ffunc=_get_ifft_axis, uafunc=_update_ifft_axes)


def __save_space_shape(a, s):
    if isinstance(a, osh5def.H5Data):
        shape = s if s is not None else a.shape
        a.data_attrs.setdefault('oshape', shape)


def __restore_space_shape(xdfunc):
    def get_shape(a, s, axes):
        if s is not None:
            return s
        if isinstance(a, osh5def.H5Data):
            return xdfunc(a, s, axes)
    return get_shape


@__restore_space_shape
def __rss_1d(a, s, axes):
    return a.data_attrs['oshape'][-1] if axes is None else a.data_attrs['oshape'][axes]


@__restore_space_shape
def __rss_2d(a, s, axes):
    return a.data_attrs['oshape'][-2:] if axes is None else tuple([a.data_attrs['oshape'][i] for i in axes])


@__restore_space_shape
def __rss_nd(a, s, axes):
    return a.data_attrs['oshape'] if axes is None else tuple([a.data_attrs['oshape'][i] for i in axes])


def rfftn(a, s=None, axes=None, norm=None):
    __save_space_shape(a, s)
    return __shifted_rfft(np.fft.rfftn)(a, s=s, axes=axes, norm=norm)


def rfft2(a, s=None, axes=(-2, -1), norm=None):
    __save_space_shape(a, s)
    return __shifted_rfft(np.fft.rfft2)(a, s=s, axes=axes, norm=norm)


def rfft(a, n=None, axis=-1, norm=None):
    __save_space_shape(a, n)
    return __shifted_rfft(np.fft.rfft)(a, s=n, axes=axis, norm=norm)


def irfftn(a, s=None, axes=None, norm=None):
    s = __rss_nd(a, s, axes)
    return __shifted_irfft(np.fft.irfftn)(a, s=s, axes=axes, norm=norm)


def irfft2(a, s=None, axes=(-2, -1), norm=None):
    s = __rss_2d(a, s, axes)
    return __shifted_irfft(np.fft.irfft2)(a, s=s, axes=axes, norm=norm)


def irfft(a, n=None, axis=-1, norm=None):
    n = __rss_1d(a, n, axis)
    return __shifted_irfft(np.fft.irfft)(a, s=n, axes=axis, norm=norm)


# # ========  Hermitian FFT  ==========
__shifted_hfft = partial(__shifted_ft_gen, sfunc=np.fft.fftshift, ffunc=np.fft.fftfreq, uafunc=_update_fft_axes)
__shifted_ihfft = partial(__shifted_ft_gen, sfunc=__idle, ffunc=_get_ihfft_axis, uafunc=_update_ifft_axes)


def hfft(a, n=None, axis=-1, norm=None):
    if n is None:
        n = a.shape[-1] if axis is None else a.shape[axis]
    nn = 2*n - 1 if n % 2 else 2*n - 2
    return __shifted_hfft(np.fft.hfft)(a, s=nn, axes=axis, norm=norm)


def ihfft(a, n=None, axis=-1, norm=None):
    return __shifted_ihfft(np.fft.ihfft)(a, s=n, axes=axis, norm=norm)
# ----------------------------------- FFT Wrappers ----------------------------------------


def field_decompose(fldarr, ffted=True, idim=None, finalize=None, outquants=('L', 't')):
    """decompose a vector field into transverse and longitudinal direction
    fldarr: list of field components in the order of x, y, z
    ffted: have the input field been
    finalize: what function to call after all transforms,
        for example finalize=abs will be converted the fields to amplitude
    idim: inverse fourier transform in idim direction(s)
    outquonts: output quantities: default=('L','t')
        'L': total amplitude square of longitudinal components
        'T': total amplitude square of transverse components
        't' or 't1', 't2', ...: transverse components, 't' means all transverse components
        'l' or 'l1', 'l2', ...: longitudinal components, 'l' means all longitudinal components
    return: list of field components in the following order (if some are not requested they will be simply omitted):
        ['L', 'T', 't', 'l']
    """
    dim = len(fldarr)
    if dim != fldarr[0].ndim:
        raise IndexError('Not enough field components for decomposition')
    if fldarr[0].ndim == 1:
        return copy.deepcopy(fldarr)
    if not finalize:
        finalize = __idle

    def wrap_up(data):
        if idim:
            return fftn(data, axes=idim)
        else:
            return data

    def rename(fld, name, longname):
        if isinstance(fld, osh5def.H5Data):
            # replace numbers in the string
            fld.name = re.sub("\d+", fld.name, name)
            fld.data_attrs['NAME'] = re.sub("\d+", name, fld.data_attrs.get('NAME', fld.name))
            fld.data_attrs['LONG_NAME'] = re.sub("\d+", longname, fld.data_attrs.get('LONG_NAME', ''))
        return fld

    if ffted:
        fftfld = [copy.deepcopy(fi) for fi in fldarr]
    else:
        fftfld = [fftn(fi) for fi in fldarr]
    kv = np.meshgrid(*reversed([x.ax for x in fftfld[0].axes]), sparse=True)
    k2 = np.sum(ki**2 for ki in kv)  # |k|^2
    k2[k2 == 0.0] = float('inf')
    kdotfld = np.divide(np.sum(np.multiply(ki, fi) for ki, fi in zip(kv, fftfld)), k2)
    fL, fT, ft, fl = 0, 0, [], []
    for i, fi in enumerate(fftfld):
        tmp = kdotfld * kv[i]
        if 't' in outquants or 't' + str(i + 1) in outquants:
            ft.append((finalize(wrap_up(fftfld[i] - tmp)), '{t' + str(i + 1) + '}'))
            if 'T' in outquants:
                fT += np.abs(ft[-1][0])**2
        elif 'T' in outquants:
            fT += np.abs(wrap_up(fftfld[i] - tmp))**2

        if 'l' in outquants or 'l'+str(i+1) in outquants:
            fl.append((finalize(wrap_up(tmp)), '{l'+str(i+1) + '}'))
            if 'L' in outquants:
                fL += np.abs(fl[-1][0])**2
        elif 'L' in outquants:
            fL += np.abs(wrap_up(tmp))**2

    res = []
    if not isinstance(fL, int):
        res.append(rename(fL, 'L', 'L^2'))
    if not isinstance(fT, int):
        res.append(rename(fT, 'T', 'T^2'))
    if ft:
        for fi in ft:
            res.append(rename(fi[0], fi[1], fi[1]))
    if fl:
        for fi in fl:
            res.append(rename(fi[0], fi[1], fi[1]))
    return tuple(res)


# modified from SciPy cookbook
def rebin(a, fac):
    """
    rebin ndarray or H5Data into a smaller ndarray or H5Data of the same rank whose dimensions
    are factors of the original dimensions. If fac in some dimension is not whole divided by
    a.shape, the residual is trimmed from the last part of array.
    example usages:
     a=rand(6,4); b=rebin(a, fac=[3,2])
     a=rand(10); b=rebin(a, fac=[3])
    """
    index = [slice(0, u - u % fac[i]) if u % fac[i] else slice(0, u) for i, u in enumerate(a.shape)]
    a = a[index]
    # update axes first
    if isinstance(a, osh5def.H5Data):
        for i, x in enumerate(a.axes):
            x.ax = x.ax[::fac[i]]

    @metasl
    def __rebin(h, fac):
        # have to convert to ndarray otherwise sum will fail
        h = h.view(np.ndarray)
        shape = h.shape
        lenShape = len(shape)
        newshape = np.floor_divide(shape, np.asarray(fac))
        evList = ['h.reshape('] + \
                 ['newshape[%d],fac[%d],'%(i,i) for i in range(lenShape)] + \
                 [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] + \
                 ['/fac[%d]'%i for i in range(lenShape)]
        return eval(''.join(evList))

    return __rebin(a, fac)


if __name__ == '__main__':
    import osh5io
    fn = 'n0-123456.h5'
    d = osh5io.read_h5(fn)
    # d = subrange(d, ((0, 35.5), (0, 166)))
    # d = np.ones((7, 20))
    # d = rebin(d, fac=[3, 3])
    # d = d.subrange(bound=[None, (None, 23., -10.)])
    d.set_value(bound=(None, (None, 140., 10.)), val=2., inverse_select=True, method=np.multiply)
    print(repr(d.view(np.ndarray)))
    c = hfft(d)
    print('c = ', repr(c))
    b = ihfft(c)
    print('b is d? ', b is d)
    diff = d - b
    print('b - d = ', diff.view(np.ndarray))
    print(repr(b))
    b = np.sqrt(b)
    # print(repr(b))

