import numpy as np
from tqdm.auto import trange
import warnings

def do_measurement(
        ts_tunneling_mus: list,  # tunneling time stamps in mus
        rec_samples: int = 500000,
        time_base_mus: float = 2.,
        clf_error_probs: tuple = (0.05, 0.05),
        start_state: bool = False,  # False = g, True = e
        start_parity: bool = False,  # False = even, True = odd
        theta: float = 1.,  # pulse for X and Y rotations, is multiplied with np.pi/2
        prob_t1_error: float = 0.,  # state decays to |g> with this probability
        prob_t2_error: float = 0.,  # state switches between |g> and |e> with this probability
):
    # simulation of the charge parity measurement through fixed-time Ramsey experiment

    # fix inputs
    ts_tunneling_mus = np.sort(ts_tunneling_mus)
    if ts_tunneling_mus.min() < 0. or ts_tunneling_mus.max() > rec_samples * time_base_mus:
        warnings.warn("Tunneling time stamps outside the measurement interval are ignored!")
        ts_tunneling_mus = ts_tunneling_mus[ts_tunneling_mus > 0.]
        ts_tunneling_mus = ts_tunneling_mus[ts_tunneling_mus < rec_samples * time_base_mus]
    print("Input: {} tunneling.".format(ts_tunneling_mus.shape[0]))

    theta *= np.pi / 2

    clf_error_probs = np.array(clf_error_probs)

    states = []
    state = start_state
    parity = start_parity

    # identify tunneling events in which sample and split into individual arrays
    tunneling_idx = np.array(ts_tunneling_mus / time_base_mus, dtype=int)
    splt_idx = np.diff(tunneling_idx).nonzero()[0] + 1
    tunneling_heres = np.split(ts_tunneling_mus, splt_idx)

    is_tunneling = np.zeros(rec_samples, dtype=bool)
    is_tunneling[np.unique(tunneling_idx)] = True

    tunnel_counter = 0

    for s in trange(rec_samples):

        # calc dphi (evolution)

        if is_tunneling[s]:
            evol_signs = np.ones(tunneling_heres[tunnel_counter].shape[0] + 1)
            evol_signs[1::2] = -1
            evol_times = np.diff(tunneling_heres[tunnel_counter],
                                 prepend=s * time_base_mus, append=(s + 1) * time_base_mus)
            dphi = np.pi / 2 * np.sum(evol_signs * evol_times) / time_base_mus
            tunnel_counter += 1
            parity = np.logical_xor(len(evol_signs) % 2 == 0, parity)
        else:
            dphi = np.pi / 2

        # calc probability densities

        # prop |g> ==> |e> (even) given by 1/2 * (1 + sin(dphi))
        # prop |g> ==> |e> (odd) given by 1/2 * (1 - sin(dphi))
        # prop |e> ==> |e> (even) given by 1/2 * (1 - sin(dphi))
        # prop |e> ==> |e> (odd) given by 1/2 * (1 + sin(dphi))

        # sign = 1-2*np.logical_xor(state, parity)
        # prob_dens = 1/2 * (1 + sign * np.sin(dphi)) 

        # prop |g> ==> |e> (even) given by 1/2 * sin^2(theta) * (1 + sin(dphi))
        # prop |g> ==> |e> (odd) given by 1/2 * sin^2(theta) * (1 - sin(dphi))
        # prop |e> ==> |e> (even) given by 1 - 1/2 * sin^2(theta) * (1 + sin(dphi))
        # prop |e> ==> |e> (odd) given by 1 - 1/2 * sin^2(theta) * (1 - sin(dphi))        

        sign = 1 - 2 * parity
        prob_dens = 1 / 2 * np.sin(theta) ** 2 * (1 + sign * np.sin(dphi))
        if state:
            prob_dens = 1 - prob_dens

        # collapse to measured state
        state = np.random.uniform(0, 1) < prob_dens

        # apply T1 error
        if state:
            state = np.random.uniform(0, 1) > prob_t1_error

        # apply T2 error
        if np.random.uniform(0, 1) < prob_t2_error:
            state = not state

        states.append(state)

    states = np.array(states).flatten()

    # classify states
    errors = np.random.uniform(0, 1, size=states.shape) < clf_error_probs[np.array(states, dtype=int)]
    asgn_state = np.logical_xor(states, errors)

    return asgn_state
