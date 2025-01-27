import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import periodogram, spectrogram, savgol_filter
from hmmlearn import hmm


def get_jumps(asg_states):
    return np.abs(np.diff(asg_states, prepend=asg_states[0])) / 2


def smooth(x, **kwargs):
    times = np.arange(0, len(x) / kwargs["smth_window"], 1) * kwargs["time_base"] * kwargs["smth_window"]
    smoothed = np.mean(x.reshape(-1, kwargs["smth_window"]), axis=-1)
    smoothed = smoothed.round()
    return times, smoothed


def get_tunnel(smoothed):
    return np.abs(np.diff(smoothed, prepend=smoothed[0]))


def fit_waiting_times(tunnel, **kwargs):
    waiting_times = np.diff(tunnel.nonzero()[0]) * kwargs["time_base"] * kwargs["smth_window"]
    hist, bins = np.histogram(waiting_times,
                              bins=np.logspace(np.log10(kwargs["time_base"] * kwargs["smth_window"]),
                                               np.log10(.0015 * kwargs["smth_window"]), 30))
    bin_centers = (bins[1:] + bins[:-1]) / 2 * 1000
    bw = bins[1:] - bins[:-1]
    idx_nzero = hist.nonzero()[0]

    def wt_ffunc(t, A1, tau1, A2, tau2):
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)

    res = curve_fit(lambda x, a, b, c, d: np.log10(wt_ffunc(x, a, b, c, d)),
                                 bin_centers[idx_nzero],
                                 np.log10(hist[idx_nzero] / bw[idx_nzero]),
                                 p0=[2e8, 0.015, 1e5, .6],
                                 bounds=([1, 1e-3, 1, 1e-3], [1e10, 1e3, 1e10, 1e3]))

    return bin_centers, hist, bw, res[0], res[1]


def psd_ffunc(freq, C, gamma_rts_1, gamma_rts_2, A, B, alpha):
    return (A * 8 * gamma_rts_1 / ((2 * gamma_rts_1) ** 2 + (2 * np.pi * freq) ** 2)
            + B * 8 * gamma_rts_2 / ((2 * gamma_rts_2) ** 2 + (2 * np.pi * freq) ** 2)
            + C * 2 * freq ** (-alpha))

def psd_ffunc_lonly(freq, A, B, C, co_0, co_1, co_2):
    return (A * 8 * co_0 / ((2 * co_0) ** 2 + (2 * np.pi * freq) ** 2) +
            B * 8 * co_1 / ((2 * co_1) ** 2 + (2 * np.pi * freq) ** 2) +
            C * 8 * co_2 / ((2 * co_2) ** 2 + (2 * np.pi * freq) ** 2))


def psd_ffunc_original(freq, t_exp, gamma_rts_1, gamma_rts_2, fidelity, coverage):
    return (fidelity ** 2 * (
                (1 - coverage) ** 2 * 8 * gamma_rts_1 / ((2 * gamma_rts_1) ** 2 + (2 * np.pi * freq) ** 2) +
                coverage ** 2 * 8 * gamma_rts_2 / ((2 * gamma_rts_2) ** 2 + (2 * np.pi * freq) ** 2)) +
            (1 - fidelity ** 2) * 2 * t_exp)

def psd_lorentzian(freq, A, gamma_rts):
    return A * 8 * gamma_rts / ((2 * gamma_rts) ** 2 + (2 * np.pi * freq) ** 2)

def psd_infidelity(freq, C, alpha):
    return C * 2 * freq ** (-alpha)


def fit_psd(jumps, p0=[2e-6, 1e2, 1e5, 0.3, 0.4, 0.25],
            filter_window=None, filter_poly=1,
            use_original=False, **kwargs):

    fs = 1 / kwargs["time_base"]

    f, Pxx_den = periodogram(jumps, fs, scaling="density")

    if filter_window is not None:
        Pxx_den = savgol_filter(Pxx_den, filter_window, filter_poly)

    if not use_original:
        res = curve_fit(
            lambda x, a, b, c, d, e, f: np.log10(psd_ffunc(freq=x, C=a, gamma_rts_1=b, gamma_rts_2=c,
                                                           A=d, B=e, alpha=f)),
            f[1:],
            np.log10(Pxx_den[1:]),
            p0=p0,
            bounds=([0., 0., 0., 0., 0., 0.], [1., 1e6, 1e6, 1., 1., 1.]))
        loss = np.sum((psd_ffunc(f[1:], *res[0]) - Pxx_den[1:]) ** 2)
    else:
        res = curve_fit(
            lambda x, b, c, d, e: np.log10(psd_ffunc_original(freq=x, t_exp=kwargs["time_base"], gamma_rts_1=b, gamma_rts_2=c,
                                                           fidelity=d, coverage=e)),
            f[1:],
            np.log10(Pxx_den[1:]),
            p0=p0[1:-1],
            bounds=([0., 0., 0., 0.], [1e6, 1e6, 1., 1.]))
        loss = np.sum((psd_ffunc_original(f[1:], kwargs["time_base"], *res[0]) - Pxx_den[1:]) ** 2)

    return f, Pxx_den, res[0], loss


def fit_psd_lonly(jumps, p0=[0.6, 0.5, 0.1, 1e2, 1e4, 5e5],
            filter_window=None, filter_poly=1, **kwargs):

    fs = 1 / kwargs["time_base"]

    f, Pxx_den = periodogram(jumps, fs, scaling="density")

    if filter_window is not None:
        Pxx_den = savgol_filter(Pxx_den, filter_window, filter_poly)

    res = curve_fit(
        lambda x, a, b, c, d, e, f: np.log10(psd_ffunc_lonly(freq=x, A=a, B=b, C=c, co_0=d, co_1=e, co_2=f)),
        f[1:],
        np.log10(Pxx_den[1:]),
        p0=p0,
        bounds=([0., 0., 0., 1., 1., 1.], [1., 1., 1., 1e6, 1e6, 1e6]))
    loss = np.sum((psd_ffunc_lonly(f[1:], *res[0]) - Pxx_den[1:]) ** 2)

    return f, Pxx_den, res[0], loss


def trigger(tunnel, expected_waiting=2.5e-5, **kwargs):
    tunnel_ts = tunnel.nonzero()[0] * kwargs["time_base"] * kwargs["smth_window"]
    gaps = np.diff(tunnel.nonzero()[0] * kwargs["time_base"] * kwargs["smth_window"], append=1e10)

    trigger_stamps = []
    duration = []
    nmbr = []
    bursts = []

    in_burst = False

    for i, (t, g) in enumerate(zip(tunnel_ts, gaps)):

        if not in_burst:
            trigger_stamps.append(t)
            in_burst = True
            burst_start = i
            bursts.append([0])
        else:
            bursts[-1].append(t - trigger_stamps[-1])

        if g > expected_waiting * kwargs["sigma"]:
            duration.append(t - trigger_stamps[-1])
            nmbr.append(i + 1 - burst_start)
            in_burst = False

    trigger_stamps = np.array(trigger_stamps)
    duration = np.array(duration)
    nmbr = np.array(nmbr)

    return trigger_stamps, duration, nmbr, bursts


def fit_tunneling(observations_sequence, rate, fidelity, pulse_period, box_size=None):
    """
    Fits a Hidden Markov Model (HMM) to an observed sequence of tunneling events, 
    extracting tunneling characteristics based on given parameters.

    Parameters
    ----------
    observations_sequence : array-like 
        The observed sequence of measurements, where each value corresponds to an observed state. When the restless measurement returns the steady/toggling readout signal, the observations_sequence is the first derivative of that, indicating where the readout state changed and where is stayed steady. 
        Typically, values are binary (e.g., `0` or `1`).

    rate : float
        The tunneling rate parameter, used to calculate the transition probabilities between states. This requires that a good guess of the tunneling rate is already available, e.g. from a PSD fit.

    fidelity : float
        The fidelity of the readout, representing the probability of correctly identifying the true state in the observed sequence. Take e.g. the average of the upper right and lower left elements of the SSRO matrix.

    pulse_period : float
        The time interval between two readout pulses.

    box_size : int, optional
        The size of the box filter to apply to the observations sequence before HMM fitting.
        If provided, a box car smoothing (low-pass filtering) is applied, with a cutoff frequency of 
        `1 / (pulse_period * box_size)`.

    Returns
    -------
    tunnel : array-like
        The extracted tunneling events derived from the predicted hidden states of the HMM.

    Notes
    -----
    - The function models the system as a two-state Hidden Markov Model (HMM) with:
        - States: ['e', 'o'] (even and odd states).
        - Observations: ['j', 's'] (jump and stay states).
    - Transition probabilities between states are calculated using the tunneling rate.
    - Emission probabilities are derived from the fidelity parameter.
    - If `box_size` is provided, the observed sequence is smoothed using a box filter before applying the HMM.
    - The HMM is solved using the Viterbi algorithm to predict the most likely hidden state sequence.

    Example
    -------
    >>> observations_sequence = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    >>> rate = 100  # Hz
    >>> fidelity = 0.95
    >>> pulse_period = 1e-6  # s
    >>> fit_tunneling(observations_sequence, rate, fidelity, pulse_period)
    """

    states = ['e', 'o']
    n_states = len(states)
    observations = ['j', 's']
    n_obs = len(observations)

    state_probability = np.array([0.5, 0.5])
    R_ = np.exp( - rate * pulse_period )  # 7951
    transition_probability = np.array([[R_, (1-R_), ],
                                       [(1-R_), R_, ],]).T

    emission_probability = np.array([[fidelity, 1-fidelity], 
                                 [1-fidelity, fidelity]])

    model = hmm.CategoricalHMM(n_components=n_states, n_features=n_obs)
    model.startprob_ = state_probability
    model.transmat_ = transition_probability
    model.emissionprob_ = emission_probability
    
    if box_size is not None:
        print('freq cutoff: ', 1/pulse_period/box_size)
        
        box_filter = np.ones(box_size) / box_size
        observations_sequence = np.convolve(observations_sequence, box_filter, mode='same')
        observations_sequence = np.array(observations_sequence > observations_sequence.mean(), dtype=int)

    hidden_states = model.predict(observations_sequence.reshape(-1,1))

    log_probability, hidden_states = model.decode(observations_sequence.reshape(-1,1), 
                                                 lengths = len(observations_sequence),
                                                 algorithm = 'viterbi')

    tunnel = get_tunnel(hidden_states)

    return pulse_period * np.arange(0,len(observations_sequence))[tunnel.nonzero()]