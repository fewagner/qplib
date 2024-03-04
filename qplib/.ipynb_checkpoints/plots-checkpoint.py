import numpy as np
import matplotlib.pyplot as plt

def plot_trace(times, smoothed, is_sim=False, show=True, **vars_dict):

    avg = np.mean(smoothed)

    fig, ax1 = plt.subplots(1,1,figsize=(6,4), dpi=300)

    plt.step(times*1000, smoothed, linewidth=1., color="purple")
    plt.axhline(avg, color='black', linestyle='dotted')

    x1 = [0,1]
    squad = ['Odd','Even']

    ax1.set_yticks(x1)
    ax1.set_yticklabels(squad, minor=False, rotation=45)

    plt.xlabel("Time (ms)")
    plt.ylabel("Charge parity estimate")
    plt.ylim(-0.3, 1.1)
    plt.xlim(350, 400)

    if is_sim:
        plt.title('Simulated poisson process')
    else:
        plt.title("Measurement time stamp: {}".format(vars_dict["msm_stamp"]))

    if show:
        plt.show()
        
        
def plot_rate(times, tunnel, lines=None, is_sim=False, show=True, **vars_dict):
    
    fig, ax1 = plt.subplots(1,1,figsize=(6,4), dpi=300)

    plt.fill_between(times[::vars_dict["binning"]]*1000, np.mean(tunnel.reshape(-1,vars_dict["binning"]), axis=-1) / (1000 * vars_dict["smth_window"] * vars_dict["time_scale"]), 
                     step='pre', color='grey', linewidth=1, alpha=0.4)
    plt.step(times[::vars_dict["binning"]]*1000, np.mean(tunnel.reshape(-1,vars_dict["binning"]), axis=-1) / (1000 * vars_dict["smth_window"] * vars_dict["time_scale"]), 
             color='grey', linewidth=1)
    plt.axhline(0., color='black', linestyle='dotted', linewidth=1)
    plt.ylabel("Tunneling rate (kHz)")
    plt.xlabel("Time (ms)")
    plt.xlim(350,400)
    plt.ylim(-1,35)
    
    for line in lines:
        plt.axvline(line, color="red")

    if is_sim:
        plt.title('Simulated poisson process')
    else:
        plt.title('Measured bursts')

    if show:
        plt.show()
        
        
def plot_waiting_times(bin_centers, hist, bw, wt_popt, is_sim=False, show=True):

    plt.figure(figsize=(6,4), dpi=300)

    plt.scatter(bin_centers, hist/bw, s=30, marker='.', color='black',
                             label='data', zorder=100)
    plt.plot(bin_centers, np.exp(-bin_centers / wt_popt[1]) * wt_popt[0], label='tau = {:.3f} ms'.format(wt_popt[1]))
    plt.plot(bin_centers, np.exp(-bin_centers / wt_popt[3]) * wt_popt[2], label='tau = {:.3f} ms'.format(wt_popt[3]))

    plt.ylim(1e2, 1e9)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Waiting time between tunneling events (ms)')
    plt.ylabel('Counts')
    plt.legend()

    # plt.text(1e-2, 1e3, 'Lines to guide the eye')

    if is_sim:
        plt.title('Simulated poisson process')
    else:
        plt.title('Measured bursts')

    if show:
        plt.show()
        
    
def plot_psd(f, Pxx_den, psd_popt, ylim=(1e-7,5), show=True, **vars_dict):
    
    plt.scatter(f, Pxx_den*f**vars_dict["alpha"], color="grey", marker='.', s=10, rasterized=True, zorder=100)
    
    plt.plot(f[1:], psd_ffunc(f[1:], *psd_popt)*f[1:], zorder=101, color='red', linewidth=2.)
    
    for factor in [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]:
        plt.plot(f, f*factor, color="green", alpha=0.1, zorder=0)
    
    plt.text(1e1,1e-6,"Time base: {:.2e} s".format(vars_dict["time_scale"]))
    
    for fq in vars_dict["freqs"]: 
        plt.axvline(fq, color='black', linestyle='dotted', linewidth=1, zorder=200)
        
    plt.text(1e0, 1e-5, 'Lines to guide the eye')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD * Frequency (arb.)')
    plt.ylim(ylim)
    
    if show:
        plt.show()