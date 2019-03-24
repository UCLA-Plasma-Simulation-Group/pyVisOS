# import osh5def
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
# try:
#     import osh5gui
#     gui_fname = osh5gui.gui_fname
# except ImportError:
#     print('Fail to import GUI routines. Check your PyQT installation')

def time_format(time=0.0, unit=None, convert_tunit=False, wavelength=0.351, **kwargs):
    if convert_tunit:
        t = wavelength * 5.31e-4 * time 
        unit = ' ps'
    else:
        t = time
    tmp = '$t = ' + "{:.2f}".format(t)
    if unit:
        tmp += '$ [$' + str(unit) + '$]'
    return tmp


def default_title(h5data, show_time=True, title=None, **kwargs):
    tmp = tex(h5data.data_attrs['LONG_NAME']) if title is None else tex(title)
    if show_time and not h5data.has_axis('t'):
        try:
            tmp += ', ' + time_format(h5data.run_attrs['TIME'][0], h5data.run_attrs['TIME UNITS'], **kwargs)
        except:  # most likely we don't have 'TIME' or 'TIME UNITS' in run_attrs
            pass
    return tmp


def tex(s):
    return '$' + s + '$' if s else ''


def axis_format(name=None, unit=None):
    s = '$' + str(name) + '$' if name else ''
    if unit:
        s += ' [$' + str(unit) + '$]'
    return s


def osplot(h5data, *args, **kwpassthrough):
    if h5data.ndim == 1:
        plot_object = osplot1d(h5data, *args, **kwpassthrough)
    elif h5data.ndim == 2:
        plot_object = osimshow(h5data, *args, **kwpassthrough)
    else:
        plot_object = None
    return plot_object


def __osplot1d(func, h5data, xlabel=None, ylabel=None, xlim=None, ylim=None, title=None, ax=None,
               convert_tunit=False, convert_xaxis=False, wavelength=0.351, transpose=False,
               *args, **kwpassthrough):
    if convert_xaxis:
        xaxis, xunit = h5data.axes[0].to_phys_unit()
    else:
        xaxis, xunit = h5data.axes[0], h5data.axes[0].attrs['UNITS']
    if ax is not None:
        set_xlim, set_ylim, set_xlabel, set_ylabel, set_title = \
            ax.set_xlim, ax.set_ylim, ax.set_xlabel, ax.set_ylabel, ax.set_title
    else:
        set_xlim, set_ylim, set_xlabel, set_ylabel, set_title = \
            plt.xlim, plt.ylim, plt.xlabel, plt.ylabel, plt.title
    if transpose:
        set_xlim, set_ylim, set_xlabel, set_ylabel = \
            set_ylim, set_xlim, set_ylabel, set_xlabel
        plot_object = func(h5data.view(np.ndarray), xaxis, *args, **kwpassthrough)
    else:
        plot_object = func(xaxis, h5data.view(np.ndarray), *args, **kwpassthrough)
    if xlabel is None:
        xlabel = axis_format(h5data.axes[0].attrs['LONG_NAME'], xunit)
    if ylabel is None:
        ylabel = axis_format(h5data.data_attrs['LONG_NAME'], str(h5data.data_attrs['UNITS']))
    _xlim = xlim or (xaxis[0], xaxis[-1])
    set_xlim(_xlim)
    if ylim is not None:
        set_ylim(ylim)
    set_xlabel(xlabel)
    set_ylabel(ylabel)
    if title is None:
        title = default_title(h5data, convert_tunit=convert_tunit, wavelength=wavelength)
    set_title(title)
    return plot_object


def osplot1d(h5data, *args, ax=None, **kwpassthrough):
    plot = plt.plot if ax is None else ax.plot
    return __osplot1d(plot, h5data, *args, **kwpassthrough)


def ossemilogx(h5data, *args, ax=None, **kwpassthrough):
    semilogx = plt.semilogx if ax is None else ax.semilogx
    return __osplot1d(semilogx, h5data, *args, **kwpassthrough)


def ossemilogy(h5data, *args, ax=None, **kwpassthrough):
    semilogy = plt.semilogy if ax is None else ax.semilogy
    return __osplot1d(semilogy, h5data, *args, **kwpassthrough)


def osloglog(h5data, *args, ax=None, **kwpassthrough):
    loglog = plt.loglog if ax is None else ax.loglog
    return __osplot1d(loglog, h5data, *args, **kwpassthrough)


def add_colorbar(im, fig=None, cax=None, ax=None, cb=None, cblabel='', use_gridspec=True, **kwargs):
    if not cb:
        cb = plt.colorbar(im, cax=cax, ax=ax, label=cblabel, **kwargs) if fig is None \
             else fig.colorbar(im, cax=cax, ax=ax, label=cblabel, **kwargs)
    else:
        cb.set_label(cblabel)
    return cb


def __osplot2d(func, h5data, *args, xlabel=None, ylabel=None, cblabel=None, title=None, xlim=None, ylim=None, clim=None,
               colorbar=True, ax=None, im=None, cb=None, convert_xaxis=False, convert_yaxis=False, fig=None,
               convert_tunit=False, wavelength=0.351, colorbar_kw=None, **kwpassthrough_plotting):
    if convert_xaxis:
        axis = h5data.axes[1].to_phys_unit(wavelength=wavelength)
        extx = axis[0].min(), axis[0].max()
        xunit = axis[1]
    else:
        extx = h5data.axes[1].ax.min(), h5data.axes[1].ax.max()
        xunit = h5data.axes[1].attrs['UNITS']

    if convert_yaxis:
        axis = h5data.axes[0].to_phys_unit(wavelength=wavelength)
        exty = axis[0].min(), axis[0].max()
        yunit = axis[1]
    else:
        exty = h5data.axes[0].ax.min(), h5data.axes[0].ax.max()
        yunit = h5data.axes[0].attrs['UNITS']
    # not a very good idea, we should have a better way to do this
    if len(args) > 0 and type(h5data) == type(args[0]):  # it is a vector field we are plotting
        fld2, if_vector_field = args[0], True
        co = kwpassthrough.pop('color', np.sqrt(h5data.values**2 + fld2.values**2) if colorbar else None)
#         vmin = kwpassthrough.pop('vmin', np.min(co))
#         vmax = kwpassthrough.pop('vmax', np.max(co))
        plot_object = func(h5data.axes[1].ax, h5data.axes[0].ax, h5data.values, 
                           fld2.values, *args[1:], color=co, **kwpassthrough_plotting)
    else:
        extent_stuff, if_vector_field = [extx[0], extx[1], exty[0], exty[1]], False
        plot_object = func(h5data.view(np.ndarray), *args, extent=extent_stuff, **kwpassthrough_plotting)

    __set_axes_labels_and_title_2d(h5data, xunit, yunit, ax=ax, xlabel=xlabel, ylabel=ylabel,
                                   convert_tunit=convert_tunit, title=title,
                                   xlim=xlim, ylim=ylim, wavelength=wavelength)
    if clim is not None:
        plot_object.set_clim(clim)

    if colorbar:
        clb = cblabel if cblabel is not None else h5data.data_attrs['UNITS'].tex()
        if colorbar_kw is None:
            colorbar_kw = {}
        ncb = add_colorbar(plot_object, fig=fig, ax=ax, cb=cb, cblabel=clb, **colorbar_kw)
        return plot_object, ncb
    return plot_object, None


def __set_axes_labels_and_title_2d(h5data, xunit, yunit, ax=None, convert_tunit=False,
                                   xlabel=None, ylabel=None, title=None,
                                   xlim=None, ylim=None, wavelength=0.351,
                                   **__unused):
    if ax is None:
        set_xlim, set_ylim, set_xlabel, set_ylabel, set_title = \
            plt.xlim, plt.ylim, plt.xlabel, plt.ylabel, plt.title
    else:
        set_xlim, set_ylim, set_xlabel, set_ylabel, set_title = \
            ax.set_xlim, ax.set_ylim, ax.set_xlabel, ax.set_ylabel, ax.set_title

    if xlim is not None:
        set_xlim(xlim)
    if ylim is not None:
        set_ylim(ylim)
    if xlabel is None:
        xlabel = axis_format(h5data.axes[1].attrs['LONG_NAME'], xunit)
    if ylabel is None:
        ylabel = axis_format(h5data.axes[0].attrs['LONG_NAME'], yunit)
    if title is None:
        title = default_title(h5data, convert_tunit=convert_tunit, wavelength=wavelength)
    if xlabel is not False:
        set_xlabel(xlabel)
    if ylabel is not False:
        set_ylabel(ylabel)
    if title is not False:
        set_title(title)


def osstreamplot(field1, field2, *args, ax=None, **kwpassthrough):
    streamplot = ax.streamplot if ax is not None else plt.streamplot
    colorbar = 'cmap' in kwpassthrough or 'color' in kwpassthrough
    return __osplot2d(streamplot, field1, field2, *args, ax=ax, colorbar=colorbar, **kwpassthrough)


def osimshow(h5data, *args, ax=None, cb=None, aspect='auto', origin='lower', **kwpassthrough):
    imshow = ax.imshow if ax is not None else plt.imshow
    return __osplot2d(imshow, h5data, *args, ax=ax, cb=cb, aspect=aspect, origin=origin, **kwpassthrough)


def osspy(h5data, *args, ax=None, aspect='auto', origin='lower', xlabel=None,
          ylabel=None, title=None, xlim=None, ylim=None, convert_tunit=False,
          wavelength=0.351, **kwpassthrough):
    spy = ax.spy if ax is not None else plt.spy
    plot_object = spy(h5data.view(np.ndarray), *args, aspect=aspect, origin=origin, **kwpassthrough)
    __set_axes_labels_and_title_2d(h5data, '', '', ax=ax, xlabel=xlabel,
                                   ylabel=ylabel, convert_tunit=convert_tunit, title=title,
                                   xlim=xlim, ylim=ylim, wavelength=wavelength)
    return plot_object


def oscontour(h5data, *args, ax=None, cb=None, **kwpassthrough):
    contour = ax.contour if ax is not None else plt.contour
    return __osplot2d(contour, h5data, *args, ax=ax, cb=cb, **kwpassthrough)


def oscontourf(h5data, *args, ax=None, cb=None, **kwpassthrough):
    contourf = ax.contourf if ax is not None else plt.contourf
    return __osplot2d(contourf, h5data, *args, ax=ax, cb=cb, **kwpassthrough)


def new_fig(h5data, *args, figsize=None, dpi=None, facecolor=None, edgecolor=None, linewidth=0.0, frameon=None,
            tight_layout=None, **kwpassthrough):
    plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, frameon=frameon,
               tight_layout=tight_layout)
    osplot(h5data, *args, **kwpassthrough)
    plt.show()

