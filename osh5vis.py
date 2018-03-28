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
        plot_object = plt.plot(h5data.axes[0].ax, h5data.view(np.ndarray), **kwpassthrough)
        plt.xlabel(axis_format(h5data.axes[0].attrs['LONG_NAME'], h5data.axes[0].attrs['UNITS']))
        plt.ylabel(axis_format(h5data.data_attrs['LONG_NAME'], str(h5data.data_attrs['UNITS'])))
    elif h5data.ndim == 2:
        extent_stuff = [h5data.axes[1].min(), h5data.axes[1].max(),
                        h5data.axes[0].min(), h5data.axes[0].max()]
        plot_object = plt.imshow(h5data, extent=extent_stuff, aspect='auto', origin='lower', **kwpassthrough)
        plt.xlabel(axis_format(h5data.axes[1].attrs['LONG_NAME'], h5data.axes[1].attrs['UNITS']))
        plt.ylabel(axis_format(h5data.axes[0].attrs['LONG_NAME'], h5data.axes[0].attrs['UNITS']))
        plt.title(tex(h5data.data_attrs['LONG_NAME']))
        cb = plt.colorbar(plot_object)
        cb.set_label(h5data.data_attrs['UNITS'].tex())
    return plot_object


def new_fig(h5data, figsize=None, dpi=None, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None,
               tight_layout=None, constrained_layout=None, **kwpassthrough):
    plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, frameon=frameon,
               tight_layout=tight_layout, constrained_layout=constrained_layout)
    osplot(h5data, **kwpassthrough)
    plt.show()

