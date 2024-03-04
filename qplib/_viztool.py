import plotly.graph_objs as go
import plotly.offline as py
import pandas as pd
import numpy as np
import ipywidgets as widgets
from ipywidgets import interactive, HBox, VBox, Button, Layout
from IPython.display import display


class VizTool():

    def __init__(self,
                 vals,
                 names,
                 bins=100,
                 *args, **kwargs):

        assert vals.shape[1] == len(names), 'vals and names must have the same length!'

        self.nmbr_features = len(vals)
        self.names = names
        self.data = {}
        for n, v in zip(names, vals.T):
            self.data[n] = v
        self.data = pd.DataFrame(self.data)

        # general
        self.data['Index'] = self.data.index
        self.N = len(self.data)
        self.bins = bins
        self.remaining_idx = np.array(list(self.data.index))
        self.color_flag = np.ones(len(self.remaining_idx))
        self.savepath = None

    def set_idx(self, remaining_idx: list):
        """
        Display only the events with these indices.

        :param remaining_idx: A list of the indices that should be displayed.
        :type remaining_idx: list
        """
        assert len(remaining_idx.shape) == 1, 'remaining_idx needs to be a list of integers!'
        remaining_idx = np.array(remaining_idx)
        try:
            self.data = self.data.loc[remaining_idx]
            self.remaining_idx = remaining_idx
            if hasattr(self, 'f0'):
                self.f0.data[0].selectedpoints = list(range(len(self.data)))
                self.sel = list(range(len(self.data)))
                self._update_axes(self.xaxis, self.yaxis)
        except:
            raise NotImplementedError('You cannot use the set_idx function anymore once you applied cuts in this method!')

    def set_colors(self, color_flag: list):
        """
        Provide a list with numerical values, that correspond to the color intensities of the events.

        :param color_flag: The color intensities of the events.
        :type color_flag: list
        """
        assert len(self.remaining_idx) == len(color_flag), 'color flag must have same length as remaining indices!'
        self.color_flag = np.array(color_flag)

    def show(self):
        """
        Start the interactive visualization.
        """
        py.init_notebook_mode()

        # scatter plot
        self.f0 = go.FigureWidget([go.Scattergl(y=self.data[self.names[0]],
                                                x=self.data[self.names[0]],
                                                mode='markers',
                                                marker_color=self.color_flag)])
        scatter = self.f0.data[0]
        self.f0.layout.xaxis.title = self.names[0]
        self.f0.layout.yaxis.title = self.names[0]
        self.xaxis = self.names[0]
        self.yaxis = self.names[0]

        scatter.marker.opacity = 0.5

        self.sel = np.arange(len(self.remaining_idx))

        # histograms
        self.f1 = go.FigureWidget([go.Histogram(x=self.data[self.names[0]], nbinsx=self.bins)])
        self.f1.layout.xaxis.title = self.names[0]
        self.f1.layout.yaxis.title = 'Counts'

        self.f2 = go.FigureWidget([go.Histogram(x=self.data[self.names[0]], nbinsx=self.bins)])
        self.f2.layout.xaxis.title = self.names[0]
        self.f2.layout.yaxis.title = 'Counts'

        # dropdown menu
        axis_dropdowns = interactive(self._update_axes, yaxis=self.data.select_dtypes('float64').columns,
                                     xaxis=self.data.select_dtypes('float64').columns)

        scatter.on_selection(self._selection_scatter_fn)

        # button for drop
        cut_button = widgets.Button(description="Cut Selected")
        cut_button.on_click(self._button_cut_fn)

        # button for linear
        linear_button = widgets.Button(description="Linear")
        linear_button.on_click(self._button_linear_fn)

        # button for log
        log_button = widgets.Button(description="Log")
        log_button.on_click(self._button_log_fn)

        self.output = widgets.Output()

        # Put everything together
        display(VBox((HBox(axis_dropdowns.children), self.f0,
                      HBox([cut_button]), self.output,
                      HBox([linear_button, log_button]),
                      self.f1, self.f2)))  # , self.t
    # private

    def _update_axes(self, xaxis, yaxis):
        self.xaxis = xaxis
        self.yaxis = yaxis
        scatter = self.f0.data[0]
        scatter.x = self.data[xaxis]
        scatter.y = self.data[yaxis]
        histx = self.f1.data[0]
        histx.x = self.data[xaxis][self.remaining_idx[self.sel]]
        histy = self.f2.data[0]
        histy.x = self.data[yaxis][self.remaining_idx[self.sel]]
        self.f0.layout.xaxis.title = xaxis
        self.f0.layout.yaxis.title = yaxis
        self.f1.layout.xaxis.title = xaxis
        self.f2.layout.xaxis.title = yaxis

    def _selection_scatter_fn(self, trace, points, selector):
        # self.t.data[0].cells.values = [self.data.loc[self.remaining_idx[points.point_inds]][col] for col in
        #                                self.table_names]
        self.f1.data[0].x = self.data[self.xaxis][self.remaining_idx[points.point_inds]]
        self.f2.data[0].x = self.data[self.yaxis][self.remaining_idx[points.point_inds]]
        self.sel = points.point_inds
        self.slider.options = self.remaining_idx[self.sel]

    def _button_cut_fn(self, b):
        if self.sel is self.data.index:
            with self.output:
                print('Select events first!')
        else:
            self.data = self.data.drop(self.remaining_idx[self.sel])
            self.remaining_idx = np.delete(self.remaining_idx, self.sel)
            self.color_flag = np.delete(self.color_flag, self.sel)
            self.f0.data[0].selectedpoints = list(range(len(self.data)))
            self.sel = list(range(len(self.data)))
            self._update_axes(self.xaxis, self.yaxis)
            with self.output:
                print('Selected events removed from dataset.')

    def _button_log_fn(self, b):
        self.f1.update_yaxes(type="log")
        self.f2.update_yaxes(type="log")

    def _button_linear_fn(self, b):
        self.f1.update_yaxes(type="linear")
        self.f2.update_yaxes(type="linear")
