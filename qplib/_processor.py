import os

import matplotlib.pyplot as plt

from ._data_tools import get_exp_data_h5, get_pulse_period
from ._ts_tools import get_jumps, psd_ffunc_lonly
from ._plots import plot_psd_lonly
from scipy.signal import periodogram, savgol_filter
from scipy.optimize import curve_fit, minimize
import numpy as np
from tqdm.auto import trange

import pdb


class QPTDataProcessor:

    def __init__(self, folders_in, folder_out, n=500000, identifier='QPT'):
        self.folders_in = folders_in
        self.folder_out = folder_out
        self.identifier = identifier
        self.files = [[fo_in + '/' + f + '/' + f + '.hdf5' for f in os.listdir(fo_in) if self.identifier in f]
                      for fo_in in self.folders_in]
        self.files = [sorted(fo) for fo in self.files]
        self.time_stamps = [[s.split('/')[-1][:6] for s in fo] for fo in self.files]
        self.qbs = [self.get_qb_names(fo) for fo in self.files]
        self.qb_names = [list({x for l in self.qbs[i] for x in l}) for i in
                         range(len(self.qbs))]
        self.n = n

        self.pulse_periods = [[get_pulse_period(f) for f in fo] for fo in self.files]
        self.time_base = np.max([np.max(p) for p in self.pulse_periods])

        self.f = np.fft.rfftfreq(self.n, self.time_base)
        self.psd = None
        self.popt = None
        self.bootstrap_files = None
        self.bootstrap_channels = None
        self.bootstrap_pperiods = None

    def get_qb_names(self, folder):
        all_qbs = []
        for f in folder:
            qbs = []
            for c in range(int(len(f.split('_')[-1][:-5]) / 4)):
                qbs.append(f.split('_')[-1][4 * (c):4 * (c + 1)])
            all_qbs.append(qbs)
        return all_qbs

    def set_qb_names(self, names):
        assert len(names) == len(self.qb_names), 'names and self.qb_names must have same len'
        for n, q in zip(names, self.qb_names):
            assert n in q, 'qubit {} is not in the corresponding folder'.format(n)
        self.names = names

    def get_asg_states(self, foidx, fidx, qidx):
        asg_states = get_exp_data_h5(self.files[foidx][fidx], channel=qidx)
        return asg_states

    def calc_psd(self,
                 starts='000000',
                 stops='235959',
                 exclude=None):

        if type(starts) == str:
            starts = [starts for _ in self.files]
        if type(stops) == str:
            stops = [stops for _ in self.files]
        assert type(starts) == list and type(stops) == list, 'starts and stops must be strings or lists'
        if exclude is None:
            exclude = [[] for _ in self.files]
        assert type(exclude) == list, 'exclude must be list or None'
        assert len(starts) == len(stops) == len(exclude) == len(self.files), \
            'lengths must match: starts, stops, exclude, folders'
        self.counter = 0.
        self.psd = np.zeros(self.f.shape[0])
        self.bootstrap_files = []
        self.bootstrap_channels = []
        self.bootstrap_pperiods = []
        # i ... idx of folder, j ... idx of file, k ... idx of qubit
        for i in trange(len(self.files)):
            for j in trange(len(self.files[i])):
                if starts[i] <= self.time_stamps[i][j] <= stops[i] and self.time_stamps[i][j] not in exclude[i]:
                    for k in range(len(self.qbs[i][j])):
                        if self.qbs[i][j][k] == self.names[i]:
                            try:
                                asg_states = get_exp_data_h5(self.files[i][j], channel=k)
                                self.bootstrap_files.append(self.files[i][j])
                                self.bootstrap_channels.append(k)

                                jumps = get_jumps(asg_states)

                                f, Pxx_den = periodogram(jumps, 1 / self.pulse_periods[i][j], scaling="density")
                                self.bootstrap_pperiods.append(self.pulse_periods[i][j])

                                # interpolate arrays to identical f grid
                                psd = np.interp(self.f, f, Pxx_den)

                                # sum up the arrays
                                self.psd += psd
                                self.counter += 1
                            except Exception as error:
                                print('File {} not processed due to {'
                                      '}'.format(self.files[i][j], error))

        self.psd /= self.counter

    def fit_psd(self,
                p0=[0.05, 0.05, 0.05, 1e2, 1e4, 1e6],
                filter_window=None,
                filter_poly=1,
                methode='ml',
                lb=[0., 0., 0., 1., 1., 1.],
                ub=[1., 1., 1., 1e7, 1e7, 1e7],
                ):
        assert self.f is not None and self.psd is not None, "should calc_psd first!"
        assert methode in ['mse', 'ml'], "methode not supported, use ml or mse"

        if filter_window is not None:
            self.filtered_psd = savgol_filter(self.psd, filter_window, filter_poly)
        else:
            self.filtered_psd = self.psd

        self.p0 = p0
        self.lb = lb
        self.ub = ub
        self.filter_window = filter_window
        self.filter_poly = filter_poly

        if methode == 'mse':
            self.res = curve_fit(
                lambda x, a, b, c, d, e, f: np.log10(psd_ffunc_lonly(freq=x, A=a, B=b, C=c, co_0=d, co_1=e, co_2=f)),
                self.f[1:],
                np.log10(self.psd[1:]),
                p0=p0,
                bounds=(lb, ub),
            )
            self.popt = self.res[0]

        elif methode == 'ml':
            def neglnlike(params, x, y, lb, ub):
                model = psd_ffunc_lonly(x, *params)
                output = np.sum(np.log(model) + y / model)

                if not np.isfinite(output) or np.any(params < lb) or np.any(params > ub):
                    return 1.0e30
                return output

            meth = 'Nelder-Mead'
            opts = {'maxfev': 3000}  # , 'xtol': 1e-16, 'ftol': 1e-16

            self.res = minimize(neglnlike,
                                np.array(p0),
                                args=(self.f, self.psd, lb, ub),
                                method=meth,
                                options=opts,
                                )
            self.popt = self.res.x

        self.loss = np.sum((np.log10(psd_ffunc_lonly(self.f[1:], *self.popt)) -
                            np.log10(self.psd[1:])) ** 2)

    def plot_psd(self, filtered=True, ylim=None, show=True):
        assert self.f is not None and self.psd is not None, "should calc_psd first!"
        assert self.popt is not None, "should fit_psd first!"

        out = plot_psd_lonly(self.f,
                             self.filtered_psd if filtered else self.psd,
                             self.popt, show=show, ylim=ylim,
                             time_base=self.time_base, freqs=self.popt[3:])
        return out

    def print_results(self):
        for i in range(3):
            print('{}. Lorentzian component: {}, {} Hz'.format(i + 1, self.popt[i], self.popt[i + 3]))

    def save(self):
        pass  # TODO

    # -------------------------------------------------
    # BOOTSTRAPPING
    # -------------------------------------------------

    @staticmethod
    def _fit_one_psd(f, psd,
                     p0=[0.05, 0.05, 0.05, 1e2, 1e4, 1e6],
                     filter_window=None,
                     filter_poly=1,
                     lb=[0., 0., 0., 1., 1., 1.],
                     ub=[1., 1., 1., 1e7, 1e7, 1e7],
                     ):

        if filter_window is not None:
            filtered_psd = savgol_filter(psd, filter_window, filter_poly)
        else:
            filtered_psd = psd

        res = curve_fit(
            lambda x, a, b, c, d, e, f: np.log10(psd_ffunc_lonly(freq=x, A=a, B=b, C=c, co_0=d, co_1=e, co_2=f)),
            f[1:],
            np.log10(filtered_psd[1:]),
            p0=p0,
            bounds=(lb, ub),
        )
        return res[0]

    def bootstrap(self,
                  chunksize=10,
                  use_popt=True,
                  ):
        assert self.bootstrap_files is not None, 'Calc PSD first!'

        self.popt_bootstrap = []

        for i in range(0, len(self.bootstrap_files), chunksize):

            psd = np.zeros(self.f.shape)
            counter = 0

            for path, ch, pper in zip(self.bootstrap_files[i:i + chunksize],
                                      self.bootstrap_channels[i:i + chunksize],
                                      self.bootstrap_pperiods[i:i + chunksize]):
                asg_states = get_exp_data_h5(path, channel=ch)
                jumps = get_jumps(asg_states)

                f, Pxx_den = periodogram(jumps, 1 / pper, scaling="density")

                # interpolate arrays to identical f grid
                psd_ = np.interp(self.f, f, Pxx_den)

                # sum up the arrays
                psd += psd_
                counter += 1

            psd /= counter

            popt = self._fit_one_psd(self.f, psd,
                                    p0=self.popt if use_popt else self.p0,
                                    filter_window=self.filter_window,
                                    filter_poly=self.filter_poly,
                                    lb=self.lb,
                                    ub=self.ub,
                                    )
            self.popt_bootstrap.append(popt)
