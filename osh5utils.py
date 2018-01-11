"""Provide basic operations for H5Data"""

import osh5def
import osaxis
import numpy as np
import copy
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
            saved.insert(0, (args[tp[0]].meta2dict(), tp[1]))
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
        tmp = tuple(ol) if len(ol) > 1 else ol[0]
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
        ax.insert(axis, osaxis.DataAxis(arr[0].run_attrs['TIME'],
                                        arr[-1].run_attrs['TIME'], len(arr), attrs=taxis_attrs))
    r = np.stack(arr, axis=axis)
    return osh5def.H5Data(r, md.timestamp, md.name, md.data_attrs, md.run_attrs, axes=ax)


# #----------------------------------- FFT Wrappers ----------------------------------------
# sfunc: for shifting; ffunc: for calculating frequency; ftfunc: for fft the data
#
def __idle(a, **kwargs):
    return a


def __try_update_axes(updfunc):
    def update_axes(a, idx, sfunc=__idle, ffunc=__idle):
        try:
            axes = a.axes
        except AttributeError:  # no axes found
            return
        try:
            iter(idx)
        except TypeError:
            idx = tuple(range(len(axes))) if idx is None else (idx,)
        # The data size can change due to the s (or n) keyword. We have to force axes update somehow.
        updfunc(axes, idx, a.shape, sfunc=sfunc, ffunc=ffunc)
    return update_axes


@__try_update_axes
def _update_fft_axes(axes, idx, shape, sfunc, ffunc):
    key, en = ['NAME', 'LONG_NAME', 'UNITS'], ['K(', 'K(', '1/(']
    for i in idx:
        axes[i].ax = sfunc(ffunc(shape[i], d=axes[i].increment())) * 2 * np.pi
        for k, e in zip(key, en):
            try:
                axes[i].attrs[k] = ''.join([e, axes[i].attrs[k], ')'])
            except (KeyError, AttributeError):
                pass


@__try_update_axes
def _update_ifft_axes(axes, idx, shape, **unused):
    key, en = ['NAME', 'LONG_NAME', 'UNITS'], [2, 2, 3]
    for i in idx:
        axes[i].ax = np.arange(0, 2*np.pi/axes[i].increment(), shape[i])
        for k, e in zip(key, en):
            try:
                axes[i].attrs[k] = axes[i].attrs[k][e:-1]
            except (KeyError, AttributeError):
                pass


def __fft_interface(ftfunc, sfunc):
    @metasl
    def fft_interface(a, s, axes, norm):
        # call fft and shift the result
        return sfunc(ftfunc(a, s=s, axes=axes, norm=norm), axes=axes)
    return fft_interface


def __shifted_fft_gen(ftfunc, sfunc, ffunc):
    def shifted_fft(a, s=None, axes=None, norm=None):
        o = __fft_interface(ftfunc, sfunc=sfunc)(a, s=s, axes=axes, norm=norm)
        _update_fft_axes(o, axes, sfunc=sfunc, ffunc=ffunc)
        return o
    return shifted_fft


def __ifft_interface(iftfunc, sfunc):
    @metasl
    def ifft_interface(a, s, axes, norm):
        # call ifft and shift the result
        return iftfunc(sfunc(a, axes=axes), s=s, axes=axes, norm=norm)
    return ifft_interface


def __shifted_ifft_gen(iftfunc, sfunc):
    def shifted_ifft(a, s=None, axes=None, norm=None):
        o = __ifft_interface(iftfunc, sfunc=sfunc)(a, s=s, axes=axes, norm=norm)
        _update_ifft_axes(o, axes)
        return o
    return shifted_ifft


# # ========  Normal FFT  ==========
__shifted_fft = partial(__shifted_fft_gen, sfunc=np.fft.fftshift, ffunc=np.fft.fftfreq)
__shifted_ifft = partial(__shifted_ifft_gen, sfunc=np.fft.ifftshift)

@__shifted_fft
def fftn(a, s=None, axes=None, norm=None):
    return np.fft.fftn(a, s=s, axes=axes, norm=norm)


@__shifted_fft
def fft2(a, s=None, axes=None, norm=None):
    return np.fft.fft2(a, s=s, axes=axes, norm=norm)


@__shifted_fft
def fft(a, s=None, axes=None, norm=None):
    if axes is None:
        axes = -1
    return np.fft.fft(a, n=s, axis=axes, norm=norm)


@__shifted_ifft
def ifftn(a, s=None, axes=None, norm=None):
    return np.fft.ifftn(a, s=s, axes=axes, norm=norm)


@__shifted_ifft
def ifft2(a, s=None, axes=None, norm=None):
    return np.fft.ifft2(a, s=s, axes=axes, norm=norm)


@__shifted_ifft
def ifft(a, s=None, axes=None, norm=None):
    if axes is None:
        axes = -1
    return np.fft.ifft(a, n=s, axis=axes, norm=norm)


# # ========  real FFT  ==========
__shifted_rfft = partial(__shifted_fft_gen, sfunc=__idle, ffunc=np.fft.rfftfreq)
__shifted_irfft = partial(__shifted_ifft_gen, sfunc=__idle)

@__shifted_rfft
def rfftn(a, s, axes, norm):
    return np.fft.rfftn(a, s=s, axes=axes, norm=norm)


@__shifted_rfft
def rfft2(a, s, axes, norm):
    return np.fft.rfft2(a, s=s, axes=axes, norm=norm)


@__shifted_rfft
def rfft(a, s, axes, norm):
    if axes is None:
        axes = -1
    return np.fft.rfft(a, n=s, axis=axes, norm=norm)


@__shifted_irfft
def rfftn(a, s, axes, norm):
    return np.fft.rfftn(a, s=s, axes=axes, norm=norm)


@__shifted_irfft
def rfft2(a, s, axes, norm):
    return np.fft.rfft2(a, s=s, axes=axes, norm=norm)


@__shifted_irfft
def rfft(a, s, axes, norm):
    if axes is None:
        axes = -1
    return np.fft.rfft(a, n=s, axis=axes, norm=norm)


# # ========  Hermitian FFT  ==========
@__shifted_irfft  # the axes operation is the same as real FFT
def hfft(a, s, axes, norm):
    if axes is None:
        axes = -1
    return np.fft.hfft(a, n=s, axis=axes, norm=norm)


@__shifted_irfft
def ihfft(a, s, axes, norm):
    if axes is None:
        axes = -1
    return np.fft.ihfft(a, n=s, axis=axes, norm=norm)
# ----------------------------------- FFT Wrappers ----------------------------------------


if __name__ == '__main__':
    import osh5io
    fn = 'test-123456.h5'
    d = osh5io.read_h5(fn)
    c = hfft(d)
    b = ihfft(c)
    print('b is d? ', b is d)
    e = d - b
    print('b - d = ', e.view(np.ndarray))

