import numpy as np


def get_jumps(asg_states):
    return np.abs(np.diff(asg_states, prepend=asg_states[0]))/2


def smooth(x, **vars_dict):
    times = np.arange(0,len(x)/vars_dict["smth_window"],1) * vars_dict["time_scale"] * vars_dict["smth_window"]
    smoothed = np.mean(jumps.reshape(-1, vars_dict["smth_window"]), axis=-1)
    smoothed = smoothed.round()
    return times, smoothed


def get_tunnel(smoothed):
    return np.abs(np.diff(smoothed, prepend=smoothed[0]))


def fit_waiting_times(tunnel, **vars_dict):
    waiting_times = np.diff(tunnel.nonzero()[0]) * vars_dict["time_scale"] * vars_dict["smth_window"]
    hist, bins = np.histogram(waiting_times, 
                              bins=np.logspace(np.log10(vars_dict["time_scale"] * vars_dict["smth_window"]),np.log10(.0015 * vars_dict["smth_window"]),30))
    bin_centers = (bins[1:] - bins[:-1])*1000
    bw = bins[1:] - bins[:-1]
    idx_nzero=hist.nonzero()[0]

    def wt_ffunc(t, A1, tau1, A2, tau2):
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)
    
    wt_popt, wt_pcov = curve_fit(lambda x,a,b,c,d: np.log10(wt_ffunc(x,a,b,c,d)), 
                             bin_centers[idx_nzero], 
                             np.log10(hist[idx_nzero]/bw[idx_nzero]),
                             p0=[2e8, 0.015, 1e5, .6],
                             bounds=([1, 1e-3, 1, 1e-3], [1e10, 1e0, 1e10, 1e0]))
    
    return bin_centers, hist, bw, wt_popt, wt_pcov


def psd_ffunc(freq, t_exp, gamma_rts_1, gamma_rts_2, fidelity_1, fidelity_2, alpha=0.25, A=1):
    return A*(fidelity_1**2*8*gamma_rts_1/((2*gamma_rts_1)**2 + (2*np.pi*freq)**2)
               + fidelity_2**2*8*gamma_rts_2/((2*gamma_rts_2)**2 + (2*np.pi*freq)**2)
                + (1-fidelity_1**2-fidelity_2**2)*t_exp*2*freq**(-alpha))


def fit_psd(jumps, p0=[1e-5, 1e2, 1e5, 0.3, 0.4, 0.25], **vars_dict):
    
    fs = 1/vars_dict["time_scale"]
    
    f, Pxx_den = periodogram(jumps, fs, scaling="density")
    
    psd_popt, psd_pcov = curve_fit(lambda x,a,b,c,d,e,f: np.log10(psd_ffunc(freq=x, t_exp=a, gamma_rts_1=b, gamma_rts_2=c, 
                                                                            fidelity_1=d, fidelity_2=e, alpha=f)), 
                             f[1:], 
                             np.log10(Pxx_den[1:]),
                             p0=p0,
                             bounds=([0., 0., 0., 0., 0., 0.], [1e-3, 1e6, 1e6, 1., 1.,1.]))
    
    return f, Pxx_den, psd_popt