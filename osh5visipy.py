from __future__ import print_function
from ipywidgets import interact, Layout
import ipywidgets as widgets
from IPython.display import display

import numpy as np

import osh5vis
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, PowerNorm, SymLogNorm


print("Importing osh5visipy. Please use `%matplotlib notebook' in your jupyter/ipython notebook")


class Generic2DPlotCtrl(object):
    tab_contents = ['Labels', 'Data', 'Colormaps']
    eps = 1e-40

    def __init__(self, data, slcs=(slice(None, ), ), title=None, norm=None):
        self._data, self._slcs = data, slcs
        # # # -------------------- Tab0 --------------------------
        # title
        if not title:
            title = osh5vis.default_title(data)
        self.if_reset_title = widgets.Checkbox(value=True, description='Auto')
        self.title = widgets.Text(value=title, placeholder='data', continuous_update=False,
                                  description='Title:', disabled=self.if_reset_title.value)
        # x label
        self.if_reset_xlabel = widgets.Checkbox(value=True, description='Auto')
        self.xlabel = widgets.Text(value=osh5vis.axis_format(data.axes[1].long_name, data.axes[1].units), placeholder='x',
                                   continuous_update=False,
                                   description='X label:', disabled=self.if_reset_xlabel.value)
        # y label
        self.if_reset_ylabel = widgets.Checkbox(value=True, description='Auto')
        self.ylabel = widgets.Text(value=osh5vis.axis_format(data.axes[0].long_name, data.axes[0].units), placeholder='y',
                                   continuous_update=False,
                                   description='Y label:', disabled=self.if_reset_ylabel.value)
        # colorbar
        self.if_reset_cbar = widgets.Checkbox(value=True, description='Auto')
        self.cbar = widgets.Text(value=data.units.tex(), placeholder='a.u.', continuous_update=False,
                                 description='Colorbar:', disabled=self.if_reset_cbar.value)
        
        tab0 = widgets.VBox([widgets.HBox([self.title, self.if_reset_title]),
                             widgets.HBox([self.xlabel, self.if_reset_xlabel]),
                             widgets.HBox([self.ylabel, self.if_reset_ylabel]),
                             widgets.HBox([self.cbar, self.if_reset_cbar])])

        # # # -------------------- Tab1 --------------------------
        items_layout = Layout(flex='1 1 auto', width='auto')
        # normalization
        # general parameters: vmin, vmax, clip
        self.if_vmin_auto = widgets.Checkbox(value=True, description='Auto', layout=items_layout)
        self.if_vmax_auto = widgets.Checkbox(value=True, description='Auto', layout=items_layout)
        self.vmin_wgt = widgets.FloatText(value=np.min(data), description='vmin', continuous_update=False,
                                          disabled=self.if_vmin_auto.value, layout=items_layout)
        self.vlogmin_wgt = widgets.FloatText(value=self.eps, description='vmin', continuous_update=False,
                                             disabled=self.if_vmin_auto.value, layout=items_layout)
        self.vmax_wgt = widgets.FloatText(value=np.max(data), description='vmax', continuous_update=False,
                                          disabled=self.if_vmin_auto.value, layout=items_layout)
        self.if_clip_cm = widgets.Checkbox(value=True, description='Clip', layout=items_layout)
        # PowerNorm specific
        self.gamma = widgets.FloatText(value=1, description='gamma', continuous_update=False, layout=items_layout)
        # SymLogNorm specific
        self.linthresh = widgets.FloatText(value=self.eps, description='linthresh', continuous_update=False,
                                           layout=items_layout)
        self.linscale = widgets.FloatText(value=1.0, description='linscale', continuous_update=False,
                                          layout=items_layout)

        # build the widgets tuple
        ln_wgt = (LogNorm, widgets.VBox([widgets.HBox([self.vlogmin_wgt, self.if_vmin_auto]),
                                         widgets.HBox([self.vmax_wgt, self.if_vmax_auto]), self.if_clip_cm]))
        n_wgt = (Normalize, widgets.VBox([widgets.HBox([self.vmin_wgt, self.if_vmin_auto]),
                                          widgets.HBox([self.vmax_wgt, self.if_vmax_auto]), self.if_clip_cm]))
        pn_wgt = (PowerNorm, widgets.VBox([widgets.HBox([self.vmin_wgt, self.if_vmin_auto]),
                                           widgets.HBox([self.vmax_wgt, self.if_vmax_auto]), self.if_clip_cm,
                                           self.gamma]))
        sln_wgt = (SymLogNorm, widgets.VBox(
            [widgets.HBox([self.vmin_wgt, self.if_vmin_auto]),
             widgets.HBox([self.vmax_wgt, self.if_vmax_auto]), self.if_clip_cm, self.linthresh, self.linscale]))

        # find out default value for norm_selector
        norm_avail = {'Log': ln_wgt, 'Normalize': n_wgt, 'Power': pn_wgt, 'SymLog': sln_wgt}
        self.norm_selector = widgets.Dropdown(options=norm_avail, 
                                              value=norm_avail.get(norm, n_wgt), description='Normalization')
        # additional care for LorNorm()
        self.__handle_lognorm()
        # re-plot button
        self.norm_btn_wgt = widgets.Button(description='Apply', disabled=False, tooltip='set colormap', icon='refresh')
        tab1 = self.__get_tab1()

        # # # -------------------- Tab2 --------------------------
        self.cmap_selector = widgets.Dropdown(options=sorted(c for c in plt.cm.datad if not c.endswith("_r")),
                                              value='jet', description='Colormap')
        tab2 = self.cmap_selector

        # construct the tab
        self.tab = widgets.Tab()
        self.tab.children = [tab0, tab1, tab2]
        [self.tab.set_title(i, tt) for i, tt in enumerate(self.tab_contents)]
        display(self.tab)

        # link and activate the widgets
        self.if_reset_title.observe(self.__update_title, 'value')
        self.if_reset_xlabel.observe(self.__update_xlabel, 'value')
        self.if_reset_ylabel.observe(self.__update_ylabel, 'value')
        self.if_reset_cbar.observe(self.__update_cbar, 'value')
        self.norm_btn_wgt.on_click(self.set_norm)
        self.if_vmin_auto.observe(self.__update_vmin, 'value')
        self.if_vmax_auto.observe(self.__update_vmax, 'value')
        self.norm_selector.observe(self.__update_norm_wgt, 'value')
        self.cmap_selector.observe(self.update_cmap, 'value')
        self.title.observe(self.update_title, 'value')
        self.xlabel.observe(self.update_xlabel, 'value')
        self.ylabel.observe(self.update_ylabel, 'value')
        self.cbar.observe(self.update_cbar, 'value')

        # plotting and then setting normalization colors
        self.im = self.plot_data()
        self.__set_norm()

    def update_data(self, data, slcs):
        self._data, self._slcs = data, slcs

    def redraw(self, data):
        """if the size of the data is the same we can just redraw part of figure"""
        self._data = data
        self.im.set_data(self.__pp(data))
        self.im.figure.canvas.draw()

    def update_title(self, change):
        self.im.axes.set_title(change['new'])

    def update_xlabel(self, change):
        self.im.axes.xaxis.set_label_text(change['new'])

    def update_ylabel(self, change):
        self.im.axes.yaxis.set_label_text(change['new'])

    def update_cbar(self, change):
        self.im.colorbar.set_label(change['new'])

    def update_cmap(self, change):
        self.im.set_cmap(change['new'])

    def refresh_tab_wgt(self, update_list):
        """
        the tab.children is a tuple so we have to reconstruct the whole tab widget when 
        addition/deletion of children widgets happens
        """
        tmp = self.tab.children
        newtab = [tmp[i] if not t else t for i, t in enumerate(update_list)]
        self.tab.children = tuple(newtab)

    def plot_data(self, **passthrough):
        return osh5vis.osimshow(self.__pp(self._data[self._slcs]), cmap=self.cmap_selector.value, **passthrough)

    def __get_tab1(self):
        return widgets.HBox([widgets.VBox([self.norm_selector, self.norm_selector.value[1]]), self.norm_btn_wgt])

    def __idle(self, data):
        return data

    def __handle_lognorm(self, change=None):
        if self.norm_selector.value[0] == LogNorm:
            self.__pp = np.abs
            self.vmax_wgt.value = np.max(np.abs(self._data))
            self.__assgin_valid_vmin()
        else:
            self.vmax_wgt.value = np.max(self._data)
            self.__assgin_valid_vmin()
            self.__pp = self.__idle

    def __update_norm_wgt(self, change=None):
        """update tab1 (second tab) only and prepare _log_data if necessary"""
        tmp = [None] * len(self.tab_contents)
        tmp[1] = self.__get_tab1()
        self.refresh_tab_wgt(tmp)
        self.__handle_lognorm(change)

    def set_norm(self, *args):
        # with LogNorm we are actually doing log(data), therefore we have to replot the whole thing to get correct cmap
        self.im.figure.clf()
        self.im = self.plot_data()
        # update norm
        self.__set_norm(*args)

    def __set_norm(self, *args):
        vmin = None if self.if_vmin_auto.value else self.norm_selector.value[1].children[0].children[0].value
        vmax = None if self.if_vmax_auto.value else self.vmax_wgt.value
        param = {'vmin': vmin, 'vmax': vmax, 'clip': self.if_clip_cm.value}
        if self.norm_selector.value[0] == PowerNorm:
            param['gamma'] = self.gamma.value
        if self.norm_selector.value[0] == SymLogNorm:
            param['linthresh'] = self.linthresh.value
            param['linscale'] = self.linscale.value
        self.im.set_norm(self.norm_selector.value[0](**param))

    def __assgin_valid_vmin(self, v=None):
        # if it is log scale
        if self.norm_selector.value[0] == LogNorm:
            self.vlogmin_wgt.value = self.eps if v is None or v < self.eps else v
        else:
            self.vmin_wgt = np.min(self._data) if v is None else v

    def __update_vmin(self, change):
        if self.if_vmin_auto.value:
            self.__assgin_valid_vmin()
            self.vmin_wgt.disabled = True
            self.vlogmin_wgt.disabled = True
        else:
            self.vmin_wgt.disabled = False
            self.vlogmin_wgt.disabled = False

    def __update_vmax(self, change):
        if self.if_vmax_auto.value:
            self.vmax_wgt.value = np.max(self._data)
            self.vmax_wgt.disabled = True
        else:
            self.vmax_wgt.disabled = False

    def __update_title(self, *args):
        if self.if_reset_title.value:
            self.title.value = osh5vis.default_title(self._data)
            self.title.disabled = True
        else:
            self.title.disabled = False

    def __update_xlabel(self, *args):
        if self.if_reset_xlabel.value:
            self.xlabel.value = osh5vis.axis_format(self._data.axes[1].long_name, self._data.axes[1].units)
            self.xlabel.disabled = True
        else:
            self.xlabel.disabled = False

    def __update_ylabel(self, *args):
        if self.if_reset_ylabel.value:
            self.ylabel.value = osh5vis.axis_format(self._data.axes[0].long_name, self._data.axes[0].units)
            self.ylabel.disabled = True
        else:
            self.ylabel.disabled = False

    def __update_cbar(self, *args):
        if self.if_reset_cbar.value:
            self.cbar.value = self._data.units.tex()
            self.cbar.disabled = True
        else:
            self.cbar.disabled = False


class Slicer(Generic2DPlotCtrl):
    def __init__(self, data, d=0, new_fig=True, **extra_kwargs):
        if new_fig:
            plt.figure()
        self.x, self.comp, self.data = data.shape[d] // 2, d, data
        self.slcs = self.__get_slice(d)
        self.axis_pos = widgets.FloatText(value=data.axes[self.comp][self.x],
                                          description=self.__axis_descr_format(), continuous_update=False)
        self.index_slider = widgets.IntSlider(min=0, max=self.data.shape[self.comp] - 1, step=1, description='index',
                                              value=self.data.shape[self.comp] // 2, continuous_update=False)

        self.axis_selector = widgets.Dropdown(options=list(range(data.ndim)), value=self.comp, description='axis:');
        self.axis_selector.observe(self.switch_slice_direction, 'value')
        self.index_slider.observe(self.update_slice, 'value')
        self.axis_pos.observe(self.__update_index_slider, 'value')

        super(Slicer, self).__init__(data[self.slcs], slcs=[i for i in self.slcs if not isinstance(i, int)],
                                     **extra_kwargs)
        display(self.axis_selector)
        display(self.axis_pos)
        display(self.index_slider)

    #         interact(self.update_slice, index=self.index_slider);

    def __update_index_slider(self, change):
        self.index_slider.value = round((self.axis_pos.value - self.data.axes[self.comp].min)
                                        / self.data.axes[self.comp].increment)

    def __axis_descr_format(self):
        return self.data.axes[self.comp].attrs['NAME'] + '/(' + self.data.axes[self.comp].attrs['UNITS'].tex() + ') = '

    def __get_slice(self, c):
        slcs = [slice(None)] * self.data.ndim
        slcs[c] = self.data.shape[c] // 2
        return slcs

    def switch_slice_direction(self, change):
        self.slcs, self.comp = self.__get_slice(change['new']), change['new']
        self.__update_axis_descr()
        self.index_slider.max = self.data.shape[self.comp] - 1
        self.update_data(self.data[self.slcs], slcs=[i for i in self.slcs if not isinstance(i, int)])
        self.im.figure.clf()
        self.im = self.plot_data()

    def __update_axis_value(self, *args):
        self.axis_pos.value = str(self.data.axes[self.comp][self.x])

    def __update_axis_descr(self, *args):
        self.axis_pos.description = self.__axis_descr_format()

    def update_slice(self, index):
        self.x = index['new']
        self.__update_axis_value()
        self.slcs[self.comp] = self.x
        self.redraw(self.data[self.slcs])

