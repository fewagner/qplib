import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import spectrogram, savgol_filter

from ._ts_tools import psd_ffunc, psd_ffunc_original, psd_lorentzian, psd_infidelity, psd_ffunc_lonly


def plot_trace(times, smoothed, xlim, is_sim=False, use_ax=None, show=True, **kwargs):

    if use_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    else:
        ax = use_ax

    avg = np.mean(smoothed)

    ax.step(times * 1000, smoothed, linewidth=1., color="purple")
    ax.axhline(avg, color='black', linestyle='dotted')

    x1 = [0, 1]
    squad = ['Odd', 'Even']

    ax.set_yticks(x1)
    ax.set_yticklabels(squad, minor=False, rotation=45)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Charge parity estimate")
    ax.set_ylim(-0.3, 1.1)
    ax.set_xlim(xlim)

    if is_sim:
        plt.title('Simulated poisson process')
    else:
        plt.title("Measurement time stamp: {}".format(kwargs["msm_stamp"]))

    if show:
        plt.show()

    if use_ax is not None:
        return ax


def plot_rate(times, tunnel, lines=None, xlim=None, ylim=None, is_sim=False, show=True,
              use_ax=None, **kwargs):
    if use_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    else:
        ax = use_ax

    ax.fill_between(times[::kwargs["binning"]] * 1000,
                    np.mean(tunnel.reshape(-1, kwargs["binning"]), axis=-1) / (
                            1000 * kwargs["smth_window"] * kwargs["time_base"]),
                    step='pre', linewidth=1, alpha=0.4,
                    color="grey" if not is_sim else "orange")
    ax.step(times[::kwargs["binning"]] * 1000, np.mean(tunnel.reshape(-1, kwargs["binning"]), axis=-1) / (
            1000 * kwargs["smth_window"] * kwargs["time_base"]),
            linewidth=1,
            color="grey" if not is_sim else "orange",
            label='measured data' if not is_sim else 'simulated poisson process')
    ax.axhline(0., color='black', linestyle='dotted', linewidth=1)
    ax.set_ylabel("Tunneling rate (kHz)")
    ax.set_xlabel("Time (ms)")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if lines is not None:
        for line in lines:
            ax.axvline(line, color="red")

    if is_sim:
        # ax.set_title('Simulated poisson process')
        pass
    else:
        ax.set_title('Measured bursts')

    if show:
        plt.show()

    if use_ax is not None:
        return ax


def plot_waiting_times(bin_centers, hist, bw, wt_popt, ylim=None, is_sim=False, use_ax=None, show=True):
    if use_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    else:
        ax = use_ax

    ax.scatter(bin_centers, hist / bw, s=30, marker='.', color='black',
                label='data', zorder=100)
    ax.plot(bin_centers, np.exp(-bin_centers / wt_popt[1]) * wt_popt[0],
             label='tau = {:.3f} ms'.format(wt_popt[1]))
    ax.plot(bin_centers, np.exp(-bin_centers / wt_popt[3]) * wt_popt[2],
             label='tau = {:.3f} ms'.format(wt_popt[3]))

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Waiting time between tunneling events (ms)')
    ax.set_ylabel('Counts')
    plt.legend()

    # plt.text(1e-2, 1e3, 'Lines to guide the eye')

    if is_sim:
        plt.title('Simulated poisson process')
    else:
        plt.title('Measured bursts')

    if show:
        plt.show()

    if use_ax is not None:
        return ax


def plot_psd(f, Pxx_den, psd_popt, ylim=None, use_ax=None, show=True, weight_by_freq=False,
             use_original=False, **kwargs):
    if use_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    else:
        ax = use_ax
        
    wei = f if weight_by_freq else np.ones(f.shape)

    ffunc = psd_ffunc if not use_original else psd_ffunc_original
    pars = psd_popt if not use_original else [kwargs["time_base"], *psd_popt]

    ax.scatter(f, Pxx_den * wei, color="grey", marker='.', s=10, rasterized=True, zorder=100,
               label='data (smoothed)')

    ax.plot(f[1:], ffunc(f[1:], *pars) * wei[1:],
             zorder=101, color='black', linewidth=2., label='fit (sum)')

    if not use_original:
        ax.plot(f[1:], psd_lorentzian(f[1:], pars[3], pars[1]) * wei[1:],
                zorder=101, color='C2', linewidth=2., label='IR-induced tunneling')
        ax.plot(f[1:], psd_lorentzian(f[1:], pars[4], pars[2]) * wei[1:],
                zorder=101, color='C3', linewidth=2., label='tunneling bursts')
        ax.plot(f[1:], psd_infidelity(f[1:], pars[0], pars[5]) * wei[1:],
                zorder=101, color='C0', linewidth=2., label='infidelity')

    if weight_by_freq:
        for factor in [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]:
            ax.plot(f, wei * factor, color="green", alpha=0.1, zorder=0)

    if "freqs" in kwargs:
        for fq,c in zip(kwargs["freqs"], ['green', 'red']):
            ax.axvline(fq, color=c, linestyle='dotted', linewidth=1, zorder=200)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)')
    if weight_by_freq:
        ax.set_ylabel('PSD * Frequency (arb.)')
    else:
        ax.set_ylabel('PSD of CP switches (1/Hz)')
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.title("Time base: {:.2f} $\mu$s".format(kwargs["time_base"] * 1e6))

    plt.legend(loc='lower left')
    plt.tight_layout()

    if show:
        plt.show()

    if use_ax is not None:
        return ax


def plot_psd_lonly(f, Pxx_den, psd_popt, ylim=None, use_ax=None, show=True,
             **kwargs):
    if use_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    else:
        ax = use_ax

    ffunc = psd_ffunc_lonly
    pars = psd_popt

    ax.scatter(f, Pxx_den, color="grey", marker='.', s=10, rasterized=True, zorder=100,
               label='data (smoothed)')

    ax.plot(f[1:], ffunc(f[1:], *pars),
            zorder=101, color='black', linewidth=2., label='fit (sum)')

    ax.plot(f[1:], psd_lorentzian(f[1:], pars[0], pars[3]),
            zorder=101, color='C2', linewidth=2., label='component 1')
    ax.plot(f[1:], psd_lorentzian(f[1:], pars[1], pars[4]),
            zorder=101, color='C3', linewidth=2., label='component 2')
    ax.plot(f[1:], psd_lorentzian(f[1:], pars[2], pars[5]),
            zorder=101, color='C0', linewidth=2., label='component 3')

    if "freqs" in kwargs:
        for fq, c in zip(kwargs["freqs"], ['C2', 'C3']):
            ax.axvline(fq, color=c, linestyle='dotted', linewidth=1, zorder=200)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD of CP switches (1/Hz)')
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.title("Time base: {:.2f} $\mu$s".format(kwargs["time_base"] * 1e6))

    plt.legend(loc='lower left', framealpha=1.0).set_zorder(1000)
    plt.tight_layout()

    if show:
        plt.show()

    if use_ax is not None:
        return ax


def plot_spectrogram(jumps, nperseg=5000, xlim=None, use_ax=None, show=True,
                     weight_by_freq=False, vals=None,
                     log=False,
                     **kwargs):
    if use_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    else:
        ax = use_ax

    fs = 1 / kwargs["time_base"]

    f, t, sxx = spectrogram(jumps, fs, nperseg=nperseg)
    
    wei = f if weight_by_freq else np.ones(f.shape)

    if vals is None:
        vals = (1e-4, 2e-1)

    if log:
        sxx = np.log10(sxx)
        vmin = np.log10(vals[0])
        vmax = np.log10(vals[1])
    else:
        vmin = vals[0]
        vmax = vals[1]
    ax.pcolormesh(t, f, sxx * wei.reshape(-1, 1), rasterized=True,
                   vmin=vmin, vmax=vmax)

    if "freqs" in kwargs:
        for fq in kwargs["freqs"]:
            ax.axhline(fq, color='white', linestyle='dotted', linewidth=1)
            pos = xlim[0] if xlim is not None else 0
            ax.text(pos, fq, "{:.1e}".format(fq), color="white")

    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')

    if weight_by_freq:
        plt.title('STFFT, bins multiplied by frequency, binning = {:.1f} ms'.format(nperseg * kwargs["time_base"] * 1000))
    else:
        plt.title('STFFT, binning = {:.1f} ms'.format(nperseg * kwargs["time_base"] * 1000))

    ax.set_yscale('symlog')
    ax.set_ylim(1 / nperseg / kwargs["time_base"], fs / 2)
    if xlim is not None:
        ax.set_xlim(xlim)

    if show:
        plt.show()

    if use_ax is not None:
        return ax

