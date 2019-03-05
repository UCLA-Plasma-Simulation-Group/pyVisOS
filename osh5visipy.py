from __future__ import print_function
from functools import partial
from ipywidgets import interact, Layout, Output
import ipywidgets as widgets
from IPython.display import display, FileLink, clear_output

import numpy as np

import osh5vis
import osh5io
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, PowerNorm, SymLogNorm
import threading


print("Importing osh5visipy. Please use `%matplotlib notebook' in your jupyter/ipython notebook;")
print("use `%matplotlib widget' if you are using newer version of matplotlib+jupyterlab")


def os2dplot_w(data, *args, pltfunc=osh5vis.osimshow, show=True, **kwargs):
    """
    2D plot with widgets
    :param data: 2D H5Data
    :param args: arguments passed to 2d plotting widgets. reserved for future use
    :param show: whether to show the widgets
    :param kwargs: keyword arguments passed to 2d plotting widgets. reserved for future use
    :return: if show == True return None otherwise return a list of widgets
    """
    wl = Generic2DPlotCtrl(data, *args, pltfunc=pltfunc, **kwargs).widgets_list
    if show:
        display(*wl)
    else:
        return wl


osimshow_w = partial(os2dplot_w, pltfunc=osh5vis.osimshow)
oscontour_w = partial(os2dplot_w, pltfunc=osh5vis.oscontour)
oscontourf_w = partial(os2dplot_w, pltfunc=osh5vis.oscontourf)


def slicer_w(data, *args, show=True, slider_only=False, **kwargs):
    """
    A slider for 3D data
    :param data: 3D H5Data or directory name (a string)
    :param args: arguments passed to plotting widgets. reserved for future use
    :param show: whether to show the widgets
    :param slider_only: if True only show the slider otherwise show also other plot control (aka 'the tab')
    :param kwargs: keyword arguments passed to 2d plotting widgets. reserved for future use
    :return: whatever widgets that are not shown
    """
    if isinstance(data, str):
        wl = DirSlicer(data, *args, **kwargs).widgets_list
        tab, slider = wl[0], widgets.HBox(wl[1:-1])
    else:
        wl = Slicer(data, *args, **kwargs).widgets_list
        tab, slider = wl[0], widgets.HBox(wl[1:-1])
    if show:
        if slider_only:
            display(slider, widgets.VBox(wl[-1:]))
            return tab
        else:
            display(tab, slider, widgets.VBox(wl[-1:]))
    else:
        return wl


def animation_w(data, *args, **kwargs):
    wl = Animation(data, *args, **kwargs).widgets_list
    display(widgets.VBox([wl[0], widgets.HBox(wl[1:4]), widgets.HBox(wl[4:-2]), widgets.VBox(wl[-2:])]))


class Generic2DPlotCtrl(object):
    tab_contents = ['Data', 'Labels', 'Axes', 'Overlay', 'Colorbar', 'Save', 'Figure']
    eps = 1e-40
    colormaps_available = sorted(c for c in plt.colormaps() if not c.endswith("_r"))

    def __init__(self, data, pltfunc=osh5vis.osimshow, slcs=(slice(None, ), ), title=None, norm=None,
                 fig_handle=None, time_in_title=True, **kwargs):

        self._data, self._slcs, self.im_xlt, self.time_in_title, self.pltfunc = \
        data, slcs, None, time_in_title, pltfunc
        user_cmap, show_colorbar = kwargs.pop('cmap', 'jet'), kwargs.pop('colorbar', True)
        tab = []
        # # # -------------------- Tab0 --------------------------
        items_layout = Layout(flex='1 1 auto', width='auto')
        # normalization
        # general parameters: vmin, vmax, clip
        self.if_vmin_auto = widgets.Checkbox(value=True, description='Auto', layout=items_layout)
        self.if_vmax_auto = widgets.Checkbox(value=True, description='Auto', layout=items_layout)
        self.vmin_wgt = widgets.FloatText(value=np.min(data), description='vmin:', continuous_update=False,
                                          disabled=self.if_vmin_auto.value, layout=items_layout)
        self.vlogmin_wgt = widgets.FloatText(value=self.eps, description='vmin:', continuous_update=False,
                                             disabled=self.if_vmin_auto.value, layout=items_layout)
        self.vmax_wgt = widgets.FloatText(value=np.max(data), description='vmax:', continuous_update=False,
                                          disabled=self.if_vmin_auto.value, layout=items_layout)
        self.if_clip_cm = widgets.Checkbox(value=True, description='Clip', layout=items_layout)
        # PowerNorm specific
        self.gamma = widgets.FloatText(value=1, description='gamma:', continuous_update=False, layout=items_layout)
        # SymLogNorm specific
        self.linthresh = widgets.FloatText(value=self.eps, description='linthresh:', continuous_update=False,
                                           layout=items_layout)
        self.linscale = widgets.FloatText(value=1.0, description='linscale:', continuous_update=False,
                                          layout=items_layout)

        # build the widgets tuple
        ln_wgt = (LogNorm, widgets.VBox([widgets.HBox([self.vmax_wgt, self.if_vmax_auto]),
                                         widgets.HBox([self.vlogmin_wgt, self.if_vmin_auto]), self.if_clip_cm]))
        n_wgt = (Normalize, widgets.VBox([widgets.HBox([self.vmax_wgt, self.if_vmax_auto]),
                                          widgets.HBox([self.vmin_wgt, self.if_vmin_auto]), self.if_clip_cm]))
        pn_wgt = (PowerNorm, widgets.VBox([widgets.HBox([self.vmax_wgt, self.if_vmax_auto]),
                                           widgets.HBox([self.vmin_wgt, self.if_vmin_auto]), self.if_clip_cm,
                                           self.gamma]))
        sln_wgt = (SymLogNorm, widgets.VBox(
            [widgets.HBox([self.vmax_wgt, self.if_vmax_auto]),
             widgets.HBox([self.vmin_wgt, self.if_vmin_auto]), self.if_clip_cm, self.linthresh, self.linscale]))

        # find out default value for norm_selector
        norm_avail = {'Log': ln_wgt, 'Normalize': n_wgt, 'Power': pn_wgt, 'SymLog': sln_wgt}
        self.norm_selector = widgets.Dropdown(options=norm_avail,
                                              value=norm_avail.get(norm, n_wgt), description='Normalization:')
        self.__old_norm = self.norm_selector.value
        # additional care for LorNorm()
        self.__handle_lognorm()
        # re-plot button
        self.norm_btn_wgt = widgets.Button(description='Apply', disabled=False, tooltip='Update normalization', icon='refresh')
        tab.append(self.__get_tab0())

        # # # -------------------- Tab1 --------------------------
        # title
        if not title:
            title = osh5vis.default_title(data, show_time=self.time_in_title)
        self.if_reset_title = widgets.Checkbox(value=True, description='Auto')
        self.title = widgets.Text(value=title, placeholder='data', continuous_update=False,
                                  description='Title:', disabled=self.if_reset_title.value)
        # x label
        self.if_reset_xlabel = widgets.Checkbox(value=True, description='Auto')
        self.xlabel = widgets.Text(value=osh5vis.axis_format(data.axes[1].long_name, data.axes[1].units),
                                   placeholder='x', continuous_update=False,
                                   description='X label:', disabled=self.if_reset_xlabel.value)
        # y label
        self.if_reset_ylabel = widgets.Checkbox(value=True, description='Auto')
        self.ylabel = widgets.Text(value=osh5vis.axis_format(data.axes[0].long_name, data.axes[0].units),
                                   placeholder='y', continuous_update=False,
                                   description='Y label:', disabled=self.if_reset_ylabel.value)

        tab.append(widgets.VBox([widgets.HBox([self.title, self.if_reset_title]),
                                 widgets.HBox([self.xlabel, self.if_reset_xlabel]),
                                 widgets.HBox([self.ylabel, self.if_reset_ylabel])]))

        # # # -------------------- Tab2 --------------------------
        self.setting_instructions = widgets.Label(value="Enter invalid value to reset", layout=items_layout)
        self.apply_range_btn = widgets.Button(description='Apply', disabled=False, tooltip='set range', icon='refresh')
        self.axis_lim_wgt = widgets.HBox([self.setting_instructions, self.apply_range_btn])
        # x axis
        xmin, xmax, xinc, ymin, ymax, yinc = self.__get_xy_minmax_delta()
        self.x_min_wgt = widgets.FloatText(value=xmin, description='xmin:', continuous_update=False,
                                           layout=items_layout)
        self.x_max_wgt = widgets.FloatText(value=xmax, description='xmax:', continuous_update=False,
                                           layout=items_layout)
        self.x_step_wgt = widgets.FloatText(value=xinc, continuous_update=False,
                                            description='$\Delta x$:', layout=items_layout)
        self.xaxis_lim_wgt = widgets.HBox([self.x_min_wgt, self.x_max_wgt, self.x_step_wgt])
        # y axis
        self.y_min_wgt = widgets.FloatText(value=ymin, description='ymin:', continuous_update=False,
                                           layout=items_layout)
        self.y_max_wgt = widgets.FloatText(value=ymax, description='ymax:', continuous_update=False,
                                           layout=items_layout)
        self.y_step_wgt = widgets.FloatText(value=yinc, continuous_update=False,
                                            description='$\Delta y$:', layout=items_layout)
        self.yaxis_lim_wgt = widgets.HBox([self.y_min_wgt, self.y_max_wgt, self.y_step_wgt])
        tab.append(widgets.VBox([self.axis_lim_wgt, self.xaxis_lim_wgt, self.yaxis_lim_wgt]))

        # # # -------------------- Tab3 --------------------------
#         overlay_item_layout = Layout(display='flex', flex='0 0 auto', width='200px')
        overlay_item_layout = Layout(display='flex', flex_flow='row wrap', width='auto')
        # x lineout
        self.xlineout_wgt = widgets.BoundedFloatText(value=ymin, min=ymin, max=ymax,
                                                     step=yinc, description=self.ylabel.value)
        widgets.jslink((self.xlineout_wgt, 'description'), (self.ylabel, 'value'))
        widgets.jslink((self.xlineout_wgt, 'min'), (self.y_min_wgt, 'value'))
        widgets.jslink((self.xlineout_wgt, 'max'), (self.y_max_wgt, 'value'))
        widgets.jslink((self.xlineout_wgt, 'step'), (self.y_step_wgt, 'value'))
        self.add_xlineout_btn = widgets.Button(description='Add', tooltip='Add x-lineout')
        self.xlineout_list_wgt = widgets.Box(children=[], layout=overlay_item_layout)
        self.xlineout_tab = widgets.VBox([widgets.HBox([self.xlineout_wgt, self.add_xlineout_btn]),
                                          self.xlineout_list_wgt])
        # y lineout
        self.ylineout_wgt = widgets.BoundedFloatText(value=xmin, min=xmin, max=xmax,
                                                     step=xinc, description=self.xlabel.value)
        widgets.jslink((self.ylineout_wgt, 'description'), (self.xlabel, 'value'))
        widgets.jslink((self.ylineout_wgt, 'min'), (self.x_min_wgt, 'value'))
        widgets.jslink((self.ylineout_wgt, 'max'), (self.x_max_wgt, 'value'))
        widgets.jslink((self.ylineout_wgt, 'step'), (self.x_step_wgt, 'value'))
        self.add_ylineout_btn = widgets.Button(description='Add', tooltip='Add y-lineout')
        self.ylineout_list_wgt = widgets.Box(children=[], layout=overlay_item_layout)
        self.ylineout_tab = widgets.VBox([widgets.HBox([self.ylineout_wgt, self.add_ylineout_btn]),
                                          self.ylineout_list_wgt])
        #TODO: overlay 2D plot

        self.overlaid_itmes = {}  # dict to keep track of the overlaid plots
        self.overlay = widgets.Tab(children=[self.xlineout_tab, self.ylineout_tab])
        [self.overlay.set_title(i, tt) for i, tt in enumerate(['x-lineout', 'y-lineout'])]
        tab.append(self.overlay)

        # # # -------------------- Tab4 --------------------------
        self.colorbar = widgets.Checkbox(value=show_colorbar, description='Show colorbar')
        self.cmap_selector = widgets.Dropdown(options=self.colormaps_available, value=user_cmap,
                                              description='Colormap:', disabled=not show_colorbar)
        self.cmap_reverse = widgets.Checkbox(value=False, description='Reverse', disabled=not show_colorbar)
        # colorbar
        self.if_reset_cbar = widgets.Checkbox(value=True, description='Auto', disabled=not show_colorbar)
        self.cbar = widgets.Text(value=data.units.tex(), placeholder='a.u.', continuous_update=False,
                                 description='Colorbar:', disabled=self.if_reset_cbar.value or not show_colorbar)
        tab.append(widgets.VBox([self.colorbar,
                                 widgets.HBox([self.cmap_selector, self.cmap_reverse], layout=items_layout),
                                 widgets.HBox([self.cbar, self.if_reset_cbar])], layout=items_layout))

        # # # -------------------- Tab5 --------------------------
        self.saveas = widgets.Button(description='Save current plot', tooltip='save current plot', button_style='')
        self.dlink = Output()
        self.figname = widgets.Text(value='figure.eps', description='Filename:')
        self.dpi = widgets.BoundedIntText(value=300, min=4, max=3000, description='DPI:')
        tab.append(widgets.VBox([widgets.HBox([self.figname, self.dpi], layout=items_layout),
                                 self.saveas, self.dlink], layout=items_layout))

        # # # -------------------- Tab6 --------------------------
        width, height = plt.rcParams.get('figure.figsize')
        self.figwidth = widgets.BoundedFloatText(value=width, min=0.1, step=0.01, description='Width:')
        self.figheight = widgets.BoundedFloatText(value=height, min=0.1, step=0.01, description='Height:')
        self.resize_btn = widgets.Button(description='Adjust figure', tooltip='Update figure', icon='refresh')
        tab.append(widgets.HBox([self.figwidth, self.figheight, self.resize_btn], layout=items_layout))

        # construct the tab
        self.tab = widgets.Tab()
        self.tab.children = tab
        [self.tab.set_title(i, tt) for i, tt in enumerate(self.tab_contents)]


        # link and activate the widgets
        self.if_reset_title.observe(self.__update_title, 'value')
        self.if_reset_xlabel.observe(self.__update_xlabel, 'value')
        self.if_reset_ylabel.observe(self.__update_ylabel, 'value')
        self.if_reset_cbar.observe(self.__update_cbar, 'value')
        self.norm_btn_wgt.on_click(self.update_norm)
        self.if_vmin_auto.observe(self.__update_vmin, 'value')
        self.if_vmax_auto.observe(self.__update_vmax, 'value')
        self.norm_selector.observe(self.__update_norm_wgt, 'value')
        self.cmap_selector.observe(self.update_cmap, 'value')
        self.cmap_reverse.observe(self.update_cmap, 'value')
        self.title.observe(self.update_title, 'value')
        self.xlabel.observe(self.update_xlabel, 'value')
        self.ylabel.observe(self.update_ylabel, 'value')
        self.cbar.observe(self.update_cbar, 'value')
        self.y_max_wgt.observe(self.__update_y_max, 'value')
        self.y_min_wgt.observe(self.__update_y_min, 'value')
        self.x_max_wgt.observe(self.__update_x_max, 'value')
        self.x_min_wgt.observe(self.__update_x_min, 'value')
        self.x_step_wgt.observe(self.__update_delta_x, 'value')
        self.y_step_wgt.observe(self.__update_delta_y, 'value')
        self.apply_range_btn.on_click(self.update_plot_area)
        self.figname.observe(self.__reset_save_button, 'value')
        self.saveas.on_click(self.__try_savefig)
        self.colorbar.observe(self.__toggle_colorbar, 'value')
        self.resize_btn.on_click(self.adjust_figure)
        self.add_xlineout_btn.on_click(self.__add_xlineout)
        self.add_ylineout_btn.on_click(self.__add_ylineout)

        # plotting and then setting normalization colors
        self.out_main = Output()
        self.observer_thrd, self.cb = None, None
        with self.out_main:
            self.fig = plt.figure(figsize=[width, height],
                                  constrained_layout=True) if fig_handle is None else fig_handle
            self.ax = self.fig.add_subplot(111)
            self.im, self.cb = self.plot_data()
#             plt.show()
        self.axx, self.axy, self._xlineouts, self._ylineouts = None, None, {}, {}

    @property
    def self(self):
        return self

    @property
    def widgets_list(self):
        return self.tab, self.out_main

    @property
    def widget(self):
        return widgets.VBox([self.tab, self.out_main])

    def update_data(self, data, slcs):
        self._data, self._slcs = data, slcs
        self.__update_title()
        self.__update_xlabel()
        self.__update_ylabel()

    def reset_plot_area(self):
        self.x_min_wgt.value, self.x_max_wgt.value, self.x_step_wgt.value, \
        self.y_min_wgt.value, self.y_max_wgt.value, self.y_step_wgt.value= self.__get_xy_minmax_delta()
        self.__destroy_all_xlineout()
        self.__destroy_all_ylineout()

    def redraw(self, data):
        if self.pltfunc is osh5vis.osimshow:
            "if the size of the data is the same we can just redraw part of figure"
            self._data = data
            self.im.set_data(self.__pp(data[self._slcs]))
            self.fig.canvas.draw()
        else:
            "for contour/contourf we have to do a full replot"
            self._data = data
            for col in self.im.collections:
                col.remove()
            self.__fully_replot()

    def update_title(self, change):
        self.ax.axes.set_title(change['new'])

    def update_xlabel(self, change):
        self.ax.axes.xaxis.set_label_text(change['new'])

    def update_ylabel(self, change):
        self.ax.axes.yaxis.set_label_text(change['new'])

    def update_cbar(self, change):
        self.im.colorbar.set_label(change['new'])

    def update_cmap(self, _change):
        self.im.set_cmap(self.cmap_selector.value if not self.cmap_reverse.value else self.cmap_selector.value + '_r')

    def adjust_figure(self, *_):
        with self.out_main:
            self.out_main.clear_output(wait=True)
            # this dosen't work in all scenarios. it could be a bug in matplotlib/jupyterlab
            self.fig.set_size_inches(self.figwidth.value, self.figheight.value)

    def __fully_replot(self):
        self.fig.delaxes(self.ax)
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.im, self.cb = self.plot_data()
#         self.fig.subplots_adjust()  # does not compatible with constrained_layout in Matplotlib 3.0

    def __get_xy_minmax_delta(self):
        return (round(self._data.axes[1].min, 2), round(self._data.axes[1].max, 2), round(self._data.axes[1].increment, 2),
                round(self._data.axes[0].min, 2), round(self._data.axes[0].max, 2), round(self._data.axes[0].increment, 2))

    def update_plot_area(self, *_):
        bnd = [(self.y_min_wgt.value, self.y_max_wgt.value, self.y_step_wgt.value),
               (self.x_min_wgt.value, self.x_max_wgt.value, self.x_step_wgt.value)]
        self._slcs = tuple(slice(*self._data.get_index_slice(self._data.axes[i], bd)) for i, bd in enumerate(bnd))
        #TODO: maybe we can keep some of the overlaid plots but __fully_replot will generate new axes.
        # for now delete everything for simplicity
        self.__destroy_all_xlineout()
        self.__destroy_all_ylineout()
        self.__fully_replot()

    def refresh_tab_wgt(self, update_list):
        """
        the tab.children is a tuple so we have to reconstruct the whole tab widget when
        addition/deletion of children widgets happens
        """
        tmp = self.tab.children
        newtab = [tmp[i] if not t else t for i, t in enumerate(update_list)]
        self.tab.children = tuple(newtab)

    def plot_data(self, **passthrough):
        return self.pltfunc(self.__pp(self._data[self._slcs]), cmap=self.cmap_selector.value,
                            norm=self.norm_selector.value[0](**self.__get_norm()), title=self.title.value,
                            xlabel=self.xlabel.value, ylabel=self.ylabel.value, cblabel=self.cbar.value,
                            ax=self.ax, fig=self.fig, colorbar=self.colorbar.value, **passthrough)

    def __get_tab0(self):
        return widgets.HBox([widgets.VBox([self.norm_selector, self.norm_selector.value[1]]), self.norm_btn_wgt])

    @staticmethod
    def _idle(data):
        return data

    def __update_twinx_scale(self):
        if self.norm_selector.value[0] == LogNorm:
            self.axx.set_yscale('log')
        elif self.norm_selector.value[0] == SymLogNorm:
            self.axx.set_yscale('symlog')
        else:
            self.axx.set_yscale('linear')

    def __destroy_all_xlineout(self):
        for li in self.xlineout_list_wgt.children:
            # remove lineout
            self._xlineouts[li.children[0]].remove()
            # remove widget
            li.close()
        # unregister all widgets
        self._xlineouts = {}
        self.xlineout_list_wgt.children = tuple()
        # remove axes
        self.axx.remove()

    def __remove_xlineout(self, btn):
        # unregister widget
        xlineout_wgt = self._xlineouts.pop(btn)
        xlineout = self._xlineouts.pop(xlineout_wgt.children[0])
        # remove x lineout
        xlineout.remove()
        # remove x lineout item widgets
        tmp = list(self.xlineout_list_wgt.children)
        tmp.remove(xlineout_wgt)
        self.xlineout_list_wgt.children = tuple(tmp)
        xlineout_wgt.close()
        # remove axes if all lineout is deleted
        if not self._xlineouts:
            self.axx.remove()
#         #TODO: a walkaround for a strange behavior of constrained_layout

    def __set_xlineout_color(self, color):
        self._xlineouts[color['owner']].set_color(color['new'])

    def __add_xlineout(self, *_):
        # add twinx if not exist
        if not self._xlineouts:
            self.axx = self.ax.twinx()
            self.__update_twinx_scale()
        pos = self._data.loc.label2int(0, self.xlineout_wgt.value)
        # plot
        xlineout = osh5vis.osplot1d(self.__pp(self._data[self._slcs])[pos, :], ax=self.axx, xlabel='', ylabel='', title='')[0]
        # add widgets (color picker + delete button)
        posstr = '%.2f' % self._data.axes[0][pos]
        nw = widgets.Button(description='', tooltip='delete %s lineout' % posstr, icon='times', layout=Layout(width='32px'))
        nw.on_click(self.__remove_xlineout)
        co = xlineout.get_color()
        cpk = widgets.ColorPicker(concise=False, description=posstr, value=co, layout=Layout(width='200px'))
        cpk.observe(self.__set_xlineout_color, 'value')
        lineout_wgt = widgets.HBox([cpk, nw], layout=Layout(width='250px', border='solid 1px', flex='0 0 auto'))
        self.xlineout_list_wgt.children += (lineout_wgt,)
        # register a new lineout
        self._xlineouts[nw], self._xlineouts[cpk] = lineout_wgt, xlineout

    def __update_xlineout(self):
        if self._xlineouts:
            for wgt in self.xlineout_list_wgt.children:
                pos = float(wgt.children[0].description)
                self._xlineouts[wgt.children[0]].set_ydata(self.__pp(self._data[self._slcs]).loc[pos, :])
            self.__update_twinx_scale()
            #TODO: autoscale for 'log' scale doesn't work after plotting the line, we have to do it manually
            #TDDO: a walkaround for a strange behavior of constrained_layout, should be removed in the future
            self.axx.set_ylabel('')

    def __update_twiny_scale(self):
        if self.norm_selector.value[0] == LogNorm:
            self.axy.set_xscale('log')
        elif self.norm_selector.value[0] == SymLogNorm:
            self.axy.set_xscale('symlog')
        else:
            self.axy.set_xscale('linear')

    def __destroy_all_ylineout(self):
        for li in self.ylineout_list_wgt.children:
            # remove lineout
            self._ylineouts[li.children[0]].remove()
            # remove widget
            li.close()
        # unregister all widgets
        self._ylineouts = {}
        self.ylineout_list_wgt.children = tuple()
        # remove axes
        self.axy.remove()

    def __remove_ylineout(self, btn):
        # unregister widget
        ylineout_wgt = self._ylineouts.pop(btn)
        ylineout = self._ylineouts.pop(ylineout_wgt.children[0])
        # remove x lineout
        ylineout.remove()
        # remove x lineout item widgets
        tmp = list(self.ylineout_list_wgt.children)
        tmp.remove(ylineout_wgt)
        self.ylineout_list_wgt.children = tuple(tmp)
        ylineout_wgt.close()
        # remove axes if all lineout is deleted
        if not self._ylineouts:
            self.axy.remove()

    def __set_ylineout_color(self, color):
        self._ylineouts[color['owner']].set_color(color['new'])

    def __add_ylineout(self, *_):
        # add twinx if not exist
        if not self._ylineouts:
            self.axy = self.ax.twiny()
            self.__update_twiny_scale()
        pos = self._data.loc.label2int(1, self.ylineout_wgt.value)
        # plot
        ylineout = osh5vis.osplot1d(self.__pp(self._data[self._slcs])[:, pos], ax=self.axy,
                                    xlabel='', ylabel='', title='', transpose=True)[0]
        # add widgets (color picker + delete button)
        posstr = '%.2f' % self._data.axes[1][pos]
        nw = widgets.Button(description='', tooltip='delete %s lineout' % posstr, icon='times', layout=Layout(width='32px'))
        nw.on_click(self.__remove_ylineout)
        co = ylineout.get_color()
        cpk = widgets.ColorPicker(concise=False, description=posstr, value=co, layout=Layout(width='200px'))
        cpk.observe(self.__set_ylineout_color, 'value')
        lineout_wgt = widgets.HBox([cpk, nw], layout=Layout(width='250px', border='solid 1px', flex='0 0 auto'))
        self.ylineout_list_wgt.children += (lineout_wgt,)
        # register a new lineout
        self._ylineouts[nw], self._ylineouts[cpk] = lineout_wgt, ylineout

    def __update_ylineout(self):
        if self._ylineouts:
            for wgt in self.ylineout_list_wgt.children:
                pos = float(wgt.children[0].description)
                self._ylineouts[wgt.children[0]].set_xdata(self.__pp(self._data[self._slcs]).loc[:, pos])
            self.__update_twiny_scale()
            #TODO: autoscale for 'log' scale doesn't work after plotting the line, we have to do it manually
            #TDDO: a walkaround for a strange behavior of constrained_layout, should be removed in the future
            self.axy.set_ylabel('')

    def __handle_lognorm(self):
        if self.norm_selector.value[0] == LogNorm:
            self.__pp = np.abs
#             self.vmax_wgt.value = np.max(np.abs(self._data))
            vmin, _ = self.__get_vminmax()
            self.__assgin_valid_vmin(v=vmin)
        else:
#             self.vmax_wgt.value = np.max(self._data)
#             self.__assgin_valid_vmin()
            self.__pp = self._idle

    def __update_norm_wgt(self, change):
        """update tab1 (second tab) only and prepare _log_data if necessary"""
        tmp = [None] * len(self.tab_contents)
        tmp[0] = self.__get_tab0()
        self.refresh_tab_wgt(tmp)
        self.__handle_lognorm()
        self.set_norm(change)
        self.__old_norm = change['old']

    def __get_vminmax(self):
        return (None if self.if_vmin_auto.value else self.norm_selector.value[1].children[1].children[0].value,
                None if self.if_vmax_auto.value else self.vmax_wgt.value)

    def __axis_descr_format(self, comp):
        return osh5vis.axis_format(self._data.axes[comp].long_name, self._data.axes[comp].units)

    def update_norm(self, *args):
        # only changing clim
        if self.__old_norm == self.norm_selector.value:
            vmin, vmax = self.__get_vminmax()
            self.im.set_clim([vmin, vmax])
        # norm change
        else:
            if self.cb:
                self.cb.remove()
            self.im.remove()
            self.im, self.cb = self.plot_data(im=self.im)
            self.__update_xlineout()
            self.__update_ylineout()

    def __get_norm(self):
        vmin, vmax = self.__get_vminmax()
        param = {'vmin': vmin, 'vmax': vmax, 'clip': self.if_clip_cm.value}
        if self.norm_selector.value[0] == PowerNorm:
            param['gamma'] = self.gamma.value
        if self.norm_selector.value[0] == SymLogNorm:
            param['linthresh'] = self.linthresh.value
            param['linscale'] = self.linscale.value
        return param

    def set_norm(self, *_):
        if self.cb:
            param = self.__get_norm()
            self.cb.set_norm(self.norm_selector.value[0](**param))

    def __assgin_valid_vmin(self, v=None):
        # if it is log scale
        if self.norm_selector.value[0] == LogNorm:
            self.vlogmin_wgt.value = self.eps if v is None or v < self.eps else v
        else:
            self.vmin_wgt.value = np.min(self._data) if v is None else v

    def __toggle_colorbar(self, change):
        if change['new']:
            self.cbar.disabled, self.if_reset_cbar.disabled, self.cmap_selector.disabled, \
            self.cmap_reverse.disabled = False, False, False, False
            self.__update_cbar(change)
        else:
            self.cbar.disabled, self.if_reset_cbar.disabled, self.cmap_selector.disabled, \
            self.cmap_reverse.disabled = True, True, True, True
            self.cb.remove()
        self.__fully_replot()

    def __update_vmin(self, _change):
        if self.if_vmin_auto.value:
            self.__assgin_valid_vmin()
            self.vmin_wgt.disabled = True
            self.vlogmin_wgt.disabled = True
        else:
            self.vmin_wgt.disabled = False
            self.vlogmin_wgt.disabled = False

    def __update_vmax(self, _change):
        if self.if_vmax_auto.value:
            self.vmax_wgt.value = np.max(self._data)
            self.vmax_wgt.disabled = True
        else:
            self.vmax_wgt.disabled = False

    def __update_title(self, *_):
        if self.if_reset_title.value:
            self.title.value = osh5vis.default_title(self._data, show_time=self.time_in_title)
            self.title.disabled = True
        else:
            self.title.disabled = False

    def __update_xlabel(self, *_):
        if self.if_reset_xlabel.value:
            self.xlabel.value = osh5vis.axis_format(self._data.axes[1].long_name, self._data.axes[1].units)
            self.xlabel.disabled = True
        else:
            self.xlabel.disabled = False

    def __update_ylabel(self, *_):
        if self.if_reset_ylabel.value:
            self.ylabel.value = osh5vis.axis_format(self._data.axes[0].long_name, self._data.axes[0].units)
            self.ylabel.disabled = True
        else:
            self.ylabel.disabled = False

    def __update_cbar(self, *_):
        if self.if_reset_cbar.value:
            self.cbar.value = self._data.units.tex()
            self.cbar.disabled = True
        else:
            self.cbar.disabled = False

    def __update_y_max(self, change):
        self.y_max_wgt.value = change['new'] if self.y_min_wgt.value < change['new'] < self._data.axes[0].max \
            else self._data.axes[0].max

    def __update_x_max(self, change):
        self.x_max_wgt.value = change['new'] if self.x_min_wgt.value < change['new'] < self._data.axes[1].max \
            else self._data.axes[1].max

    def __update_y_min(self, change):
        self.y_min_wgt.value = change['new'] if self._data.axes[0].min < change['new'] < self.y_max_wgt.value \
            else self._data.axes[0].min

    def __update_x_min(self, change):
        self.x_min_wgt.value = change['new'] if self._data.axes[1].min < change['new'] < self.x_max_wgt.value \
            else self._data.axes[1].min

    def __update_delta_y(self, change):
        if not (0 < round(change['new'] / self._data.axes[0].increment) <= self._data[self._slcs].shape[0]):
            self.y_step_wgt.value = self._data.axes[0].increment

    def __update_delta_x(self, change):
        if not (0 < round(change['new'] / self._data.axes[1].increment) <= self._data[self._slcs].shape[1]):
            self.x_step_wgt.value = self._data.axes[1].increment

    def __reset_save_button(self, *_):
        self.saveas.description, self.saveas.tooltip, self.saveas.button_style= \
        'Save current plot', 'save current plot', ''

    def __savefig(self):
        try:
            self.fig.savefig(self.figname.value, dpi=self.dpi.value)
#             self.dlink.clear_output(wait=True)
            with self.dlink:
                clear_output(wait=True)
                print('shift+right_click to downloaod:')
                display(FileLink(self.figname.value))
            self.__reset_save_button(0)
        except PermissionError:
            self.saveas.description, self.saveas.tooltip, self.saveas.button_style= \
                    'Permission Denied', 'please try another directory', 'danger'

    def __try_savefig(self, *_):
        pdir = os.path.abspath(os.path.dirname(self.figname.value))
        path_exist = os.path.exists(pdir)
        file_exist = os.path.exists(self.figname.value)
        if path_exist:
            if file_exist:
                if not self.saveas.button_style:
                    self.saveas.description, self.saveas.tooltip, self.saveas.button_style= \
                    'Overwirte file', 'overwrite existing file', 'warning'
                else:
                    self.__savefig()
            else:
                self.__savefig()
        else:
            if not self.saveas.button_style:
                self.saveas.description, self.saveas.tooltip, self.saveas.button_style= \
                'Create path & save', 'create non-existing path and save', 'warning'
            else:
                os.makedirs(pdir)
                self.__savefig()


class Slicer(Generic2DPlotCtrl):
    def __init__(self, data, d=0, **extra_kwargs):
        self.x, self.comp, self.data = data.shape[d] // 2, d, data
        self.slcs = self.__get_slice(d)
        self.axis_pos = widgets.FloatText(value=data.axes[self.comp][self.x],
                                          description=self.__axis_format(), continuous_update=False)
        self.index_slider = widgets.IntSlider(min=0, max=self.data.shape[self.comp] - 1, step=1, description='index:',
                                              value=self.data.shape[self.comp] // 2, continuous_update=False)

        self.axis_selector = widgets.Dropdown(options=list(range(data.ndim)), value=self.comp, description='axis:')
        self.axis_selector.observe(self.switch_slice_direction, 'value')
        self.index_slider.observe(self.update_slice, 'value')
        self.axis_pos.observe(self.__update_index_slider, 'value')

        super(Slicer, self).__init__(data[self.slcs], slcs=[i for i in self.slcs if not isinstance(i, int)],
                                     time_in_title=not data.has_axis('t'), **extra_kwargs)

    @property
    def widgets_list(self):
        return self.tab, self.axis_pos, self.index_slider, self.axis_selector, self.out_main

    @property
    def widget(self):
        return widgets.VBox([widgets.HBox([self.axis_pos, self.index_slider, self.axis_selector]),
                             self.out_main])

    def __update_index_slider(self, _change):
        self.index_slider.value = round((self.axis_pos.value - self.data.axes[self.comp].min)
                                        / self.data.axes[self.comp].increment)

    def __axis_format(self):
        return osh5vis.axis_format(self.data.axes[self.comp].long_name, self.data.axes[self.comp].units)

    def __get_slice(self, c):
        slcs = [slice(None)] * self.data.ndim
        slcs[c] = self.data.shape[c] // 2
        return slcs

    def switch_slice_direction(self, change):
        self.slcs, self.comp, self.x = \
            self.__get_slice(change['new']), change['new'], self.data.shape[change['new']] // 2
        self.reset_slider_index()
        self.__update_axis_descr()
        self.update_data(self.data[self.slcs], slcs=[i for i in self.slcs if not isinstance(i, int)])
        self.reset_plot_area()
        self.set_norm(change)
        if self.cb:
            self.cb.remove()
        # the following is an exact copy of __fully_replot;
        # however, calling the function wouldn't work for some reason
        self.fig.delaxes(self.ax)
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.im, self.cb = self.plot_data()
#         self.fig.subplots_adjust()  # does not compatible with constrained_layout in Matplotlib 3.0

    def reset_slider_index(self):
        # stop the observe while updating values
        self.index_slider.unobserve(self.update_slice, 'value')
        self.index_slider.max = self.data.shape[self.comp] - 1
        self.__update_axis_value()
        self.index_slider.observe(self.update_slice, 'value')

    def __update_axis_value(self, *_):
        self.axis_pos.value = str(self.data.axes[self.comp][self.x])

    def __update_axis_descr(self, *_):
        self.axis_pos.description = self.__axis_format()

    def update_slice(self, index):
        self.x = index['new']
        self.__update_axis_value()
        self.slcs[self.comp] = self.x
        self.redraw(self.data[self.slcs])


class DirSlicer(Generic2DPlotCtrl):
    def __init__(self, filefilter, processing=Generic2DPlotCtrl._idle, **extra_kwargs):
        fp = filefilter + '/*.h5' if os.path.isdir(filefilter) else filefilter
        self.filter, self.flist, self.processing = fp, sorted(glob.glob(fp)), processing
        try:
            self.data = processing(osh5io.read_h5(self.flist[0]))
        except IndexError:
            raise IOError('No file found matching ' + fp)

        items_layout = Layout(flex='1 1 auto', width='auto')
        self.file_slider = widgets.SelectionSlider(options=self.flist, description='filename:', value=self.flist[0],
                                                   continuous_update=False, layout=items_layout)
        self.time_label = widgets.Label(value=osh5vis.time_format(self.data.run_attrs['TIME'][0],
                                                                  self.data.run_attrs['TIME UNITS']),
                                        layout=items_layout)
        self.file_slider.observe(self.update_slice, 'value')

        super(DirSlicer, self).__init__(self.data, time_in_title=False, **extra_kwargs)


    @property
    def widgets_list(self):
        return self.tab, self.file_slider, self.time_label, self.out_main

    @property
    def widget(self):
        return widgets.VBox([widgets.HBox[self.file_slider, self.time_label], self.out_main])

    def update_slice(self, change):
        self.data = self.processing(osh5io.read_h5(change['new']))
        self.time_label.value = osh5vis.time_format(self.data.run_attrs['TIME'][0], self.data.run_attrs['TIME UNITS'])
        self.redraw(self.data)


class Animation(Slicer):
    def __init__(self, data, interval=10, step=1, **kwargs):
        super(Animation, self).__init__(data, **kwargs)
        self.play = widgets.Play(interval=interval, value=self.x, min=0, max=len(self.data.axes[self.comp]),
                                 step=step, description="Press play", disabled=False)
        self.interval_wgt = widgets.IntText(value=interval, description='Interval:', disabled=False)
        self.step_wgt = widgets.IntText(value=step, description='Step:', disabled=False)

        # link everything together
        widgets.jslink((self.play, 'value'), (self.index_slider, 'value'))
        self.interval_wgt.observe(self.update_interval, 'value')
        self.step_wgt.observe(self.update_step, 'value')

    @property
    def widgets_list(self):
        return (self.tab, self.axis_pos, self.index_slider, self.axis_selector,
                self.play, self.interval_wgt, self.step_wgt, self.out_main)

    def switch_slice_direction(self, change):
        super(Animation, self).switch_slice_direction(change)
        self.play.max = len(self.data.axes[self.comp])

    def update_interval(self, change):
        self.play.interval = change['new']

    def update_step(self, change):
        self.play.step = change['new']
