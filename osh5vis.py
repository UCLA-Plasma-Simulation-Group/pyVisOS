# import osh5def
import matplotlib.pyplot as plt
import numpy as np
try:
    import osh5gui
    gui_fname = osh5gui.gui_fname
except ImportError:
    print('Fail to import GUI routines. Check your PyQT installation')


def tex(s):
    return ''.join(['$', s, '$']) if s else ''


def axis_format(name=None, unit=None):
    s = '$' + str(name) + '$' if name else ''
    if unit:
        s += ' [$' + str(unit) + '$]'
    return s


def osplot(h5data, **kwpassthrough):
    if h5data.ndim == 1:
        plot_object = osplot1d(h5data, **kwpassthrough)
    elif h5data.ndim == 2:
        plot_object = osimshow(h5data, **kwpassthrough)
    else:
        plot_object = None
    return plot_object


def __osplot1d(func, h5data, **kwpassthrough):
    plot_object = func(h5data.axes[0].ax, h5data.view(np.ndarray), **kwpassthrough)
    plt.xlabel(axis_format(h5data.axes[0].attrs['LONG_NAME'], h5data.axes[0].attrs['UNITS']))
    plt.ylabel(axis_format(h5data.data_attrs['LONG_NAME'], str(h5data.data_attrs['UNITS'])))
    return plot_object


def osplot1d(h5data, **kwpassthrough):
    return __osplot1d(plt.plot, h5data, **kwpassthrough)


def ossemilogx(h5data, **kwpassthrough):
    return __osplot1d(plt.semilogx, h5data, **kwpassthrough)


def ossemilogy(h5data, **kwpassthrough):
    return __osplot1d(plt.semilogy, h5data, **kwpassthrough)


def osloglog(h5data, **kwpassthrough):
    return __osplot1d(plt.loglog, h5data, **kwpassthrough)


def __osplot2d(func, h5data, **kwpassthrough):
    extent_stuff = [h5data.axes[1].min, h5data.axes[1].max,
                    h5data.axes[0].min, h5data.axes[0].max]
    plot_object = func(h5data.view(np.ndarray), extent=extent_stuff, aspect='auto', origin='lower', **kwpassthrough)
    plt.xlabel(axis_format(h5data.axes[1].attrs['LONG_NAME'], h5data.axes[1].attrs['UNITS']))
    plt.ylabel(axis_format(h5data.axes[0].attrs['LONG_NAME'], h5data.axes[0].attrs['UNITS']))
    plt.title(tex(h5data.data_attrs['LONG_NAME']))
    cb = plt.colorbar(plot_object)
    cb.set_label(h5data.data_attrs['UNITS'].tex())
    return plot_object


def osimshow(h5data, **kwpassthrough):
    return __osplot2d(plt.imshow, h5data, **kwpassthrough)


def oscontour(h5data, **kwpassthrough):
    return __osplot2d(plt.contour, h5data, **kwpassthrough)


def oscontourf(h5data, **kwpassthrough):
    return __osplot2d(plt.contourf, h5data, **kwpassthrough)


def new_fig(h5data, figsize=None, dpi=None, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None,
            tight_layout=None, **kwpassthrough):
    plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, frameon=frameon,
               tight_layout=tight_layout)
    osplot(h5data, **kwpassthrough)
    plt.show()

