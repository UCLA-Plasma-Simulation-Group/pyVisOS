# import osh5def
import matplotlib.pyplot as plt
import numpy as np
try:
    import osh5gui
    gui_fname = osh5gui.gui_fname
except ImportError:
    print('Fail to import GUI routines. Check your PyQT installation')


def default_title(h5data):
    return ''.join([tex(h5data.data_attrs['LONG_NAME']), ', $t = ', "{:.2f}".format(h5data.run_attrs['TIME'][0]),
                    '$  [$', h5data.run_attrs['TIME UNITS'], '$]'])


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


def __osplot1d(func, h5data, xlabel=None, ylabel=None, xlim=None, ylim=None, title=None, ax=None, **kwpassthrough):
    plot_object = func(h5data.axes[0].ax, h5data.view(np.ndarray), **kwpassthrough)
    if ax is not None:
        set_xlim, set_ylim, set_xlabel, set_ylabel, set_title = \
            ax.set_xlim, ax.set_ylim, ax.set_xlabel, ax.set_ylabel, ax.set_title
    else:
        set_xlim, set_ylim, set_xlabel, set_ylabel, set_title = \
            plt.xlim, plt.ylim, plt.xlabel, plt.ylabel, plt.title
    if xlabel is None:
        xlabel = axis_format(h5data.axes[0].attrs['LONG_NAME'], h5data.axes[0].attrs['UNITS'])
    if ylabel is None:
        ylabel = axis_format(h5data.data_attrs['LONG_NAME'], str(h5data.data_attrs['UNITS']))
    if xlim is not None:
        set_xlim(xlim)
    if ylim is not None:
        set_ylim(ylim)
    set_xlabel(xlabel)
    set_ylabel(ylabel)
    if title is None:
        title = default_title(h5data)
    set_title(title)
    return plot_object


def osplot1d(h5data, ax=None, **kwpassthrough):
    plot = plt.plot if ax is None else ax.plot
    return __osplot1d(plot, h5data, **kwpassthrough)


def ossemilogx(h5data, ax=None, **kwpassthrough):
    semilogx = plt.semilogx if ax is None else ax.semilogx
    return __osplot1d(semilogx, h5data, **kwpassthrough)


def ossemilogy(h5data, ax=None, **kwpassthrough):
    semilogy = plt.semilogy if ax is None else ax.semilogy
    return __osplot1d(semilogy, h5data, **kwpassthrough)


def osloglog(h5data, ax=None, **kwpassthrough):
    loglog = plt.loglog if ax is None else ax.loglog
    return __osplot1d(loglog, h5data, **kwpassthrough)


def __osplot2d(func, h5data, xlabel=None, ylabel=None, cblabel=None, title=None, xlim=None, ylim=None, clim=None,
               colorbar=True, **kwpassthrough):
    extent_stuff = [h5data.axes[1].min, h5data.axes[1].max,
                    h5data.axes[0].min, h5data.axes[0].max]
    plot_object = func(h5data.view(np.ndarray), extent=extent_stuff, aspect='auto', origin='lower', **kwpassthrough)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xlabel is None:
        xlabel = axis_format(h5data.axes[1].attrs['LONG_NAME'], h5data.axes[1].attrs['UNITS'])
    if ylabel is None:
        ylabel = axis_format(h5data.axes[0].attrs['LONG_NAME'], h5data.axes[0].attrs['UNITS'])
    if title is None:
        title = default_title(h5data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if clim is not None:
        plt.clim(clim)
    if colorbar:
        cb = plt.colorbar(plot_object)
        if cblabel is None:
            cb.set_label(h5data.data_attrs['UNITS'].tex())
        else:
            cb.set_label(cblabel)
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

