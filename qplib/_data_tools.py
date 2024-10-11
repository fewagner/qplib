import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

from ._ts_tools import get_jumps, fit_psd_lonly
from ._plots import plot_psd_lonly

def get_exp_data_h5(fpath, channel=0):
    with h5py.File(fpath, "r") as f:
        data = f['Experimental Data/Data']
        asg_states = np.array(data[:, 1 + channel * 2])
    return asg_states.round() * 2 - 1

def get_pulse_period(fpath):
    with h5py.File(fpath, "r") as f:
        pulse_period = float(f['Instrument settings/TriggerDevice'].attrs['pulse_period'][:])
    return pulse_period

def get_ge_freq(fpath, qubit):
    with h5py.File(fpath, "r") as f:
        ge_freq = float(f['Instrument settings/' + qubit].attrs[
                      'ge_freq'][:])
    return ge_freq

def get_ro_freq(fpath, qubit):
    with h5py.File(fpath, "r") as f:
        ro_freq = float(f['Instrument settings/' + qubit].attrs[
                      'ro_freq'][:])
    return ro_freq

def process_data(data, **kwargs):
    if not "asg_states" in kwargs:
        if kwargs["prim_key"] is None:
            asg_states = data[kwargs["use_qubit"]][kwargs["msm_stamp"]]["assigned_states"]
            kwargs["time_base"] = data[kwargs["use_qubit"]][kwargs["msm_stamp"]]["pulse_period"]
        else:
            asg_states = data[kwargs["prim_key"]][kwargs["use_qubit"]][kwargs["msm_stamp"]]["assigned_states"]
            kwargs["time_base"] = data[kwargs["prim_key"]][kwargs["use_qubit"]][kwargs["msm_stamp"]]["pulse_period"]
    else:
        asg_states = kwargs["asg_states"]
        if not "time_base" in kwargs:
            raise Exception("When providing asg_states, also time_base must be provided.")

    jumps = get_jumps(asg_states)

    # --------------------------------------------------------
    # fit the PSD
    # --------------------------------------------------------

    f, Pxx_den, psd_popt, loss = fit_psd_lonly(jumps,
                                               filter_window=20,
                                               filter_poly=1,
                                               **kwargs)

    # --------------------------------------------------------
    # save PSD plot
    # --------------------------------------------------------

    if kwargs["plot"]:
        plot_psd_lonly(f, Pxx_den, psd_popt, ylim=(1e-9, 1e-1),
                       filter_window=100,
                       filter_poly=1,
                       show=False,
                       **kwargs)

        plt.savefig(
            kwargs["path_out"] + "/" + kwargs["path_pkl"].split("/")[-1][:-4] + "_" + kwargs["use_qubit"] + "_" +
            kwargs["msm_stamp"] + "_psd" + kwargs["name_app"] + ".png", dpi=300)
        plt.close()

    # --------------------------------------------------------
    # write fitted parameters to a txt file
    # --------------------------------------------------------

    np.save(
        kwargs["path_out"] + "/" + kwargs["path_pkl"].split("/")[-1][:-4] + "_" + kwargs["use_qubit"] + "_" + kwargs[
            "msm_stamp"] + "_popt" + kwargs["name_app"] + ".npy", psd_popt)

    par_names = ["A", "B", "C", "co_0", "co_1", "co_2"]

    with open(kwargs["path_out"] + "/" + kwargs["path_pkl"].split("/")[-1][:-4] + "_" + kwargs["use_qubit"] + "_" +
              kwargs["msm_stamp"] + "_psd" + kwargs["name_app"] + ".txt", 'w') as f:
        f.write("time base\t{}".format(kwargs["time_base"]))
        f.write('\n')
        for n, v in zip(par_names, psd_popt):
            f.write("{}\t{}".format(n, v))
            f.write('\n')
        f.write("loss\t{}".format(loss))

def difference_strings(string1, string2):
    string1 = string1.split('_')
    string2 = string2.split('_')
    A = set(string1)
    B = set(string2)
    return sorted(list(A.symmetric_difference(B)))


def load_processed_data(dir: str, days: list, time_los: list, time_his: list, qubits: list, **kwargs):
    if 'cooldown_dates' in kwargs:
        with open(kwargs["cooldown_dates"], "r") as file:
            dates = [datetime.strptime(d, "%Y%m%d") for d in file.read().split('\n')]

    vals = []
    names = []
    count = 0

    for d_, tl_, th_, q_ in zip(days, time_los, time_his, qubits):

        # read file names in dir
        # consider only the txt files
        fnames = [fn for fn in os.listdir(dir) if fn[-4:] == '.txt']

        # consider only files with right day and qubit in name
        fnames = [fn for fn in fnames if np.logical_and(d_ in fn, q_ in fn)]
        fnames.sort()

        # get all time stamps
        tstamps = []
        tstamps.extend(difference_strings(fnames[0], fnames[1]))
        for fn in fnames[2:]:
            diff = difference_strings(fnames[0], fn)
            tstamps.append(diff[1])

        # make list for all vals
        these_vals = []

        # go through the time stamps and use only those in the interval
        for fn, ts in zip(fnames, tstamps):
            if tl_ is not None:
                after_start = int(ts) >= int(tl_)
            else:
                after_start = True
            if th_ is not None:
                before_end = int(ts) <= int(th_)
            else:
                before_end = True
            if np.logical_and(after_start, before_end):
                count += 1

                # read the txt file
                with open(dir + '/' + fn, "r") as file:
                    file_content = file.read()

                    # get names from txt file
                    if names == []:
                        names = [line.split('\t')[0] for line in file_content.split('\n')]

                    # seperate the lines
                    # put the values in a list as floats
                    # content = [[line.split('\t')[0], float(line.split('\t')[1])] for line in file.read().split('\n')]
                    content = [float(line.split('\t')[1]) for line in file_content.split('\n')]

                    if 'cooldown_dates' in kwargs:
                        this_date = datetime.strptime(d_, "%Y%m%d") + timedelta(hours=int(ts[:2]), minutes=int(ts[2:4]),
                                                                                seconds=int(ts[4:]))
                        deltas = [this_date - d for d in dates if this_date > d]
                        min_deltas = np.min(deltas)
                        content.append(min_deltas.days * 24 + min_deltas.seconds / 3600)
                    else:
                        content.append(-1.)

                    # append to long list
                    these_vals.append(content)

        vals.extend(these_vals)
    print('Loaded {} files.'.format(count))

    # turn long list in numpy array
    vals = np.array(vals)

    if 'cooldown_dates' in kwargs:
        names.append("hours_since_cooldown")

    return vals, names


