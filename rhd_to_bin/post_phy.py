import os
import math
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
# fallback to cmd line prompts when gui not available
try:
    gui = True
    import tkinter as tk
    from tkinter import filedialog
    gui_root = tk.Tk()
    # windows ('nt') vs linux
    if os.name == 'nt':
        gui_root.attributes('-topmost', True, '-alpha', 0)
    else:
        gui_root.withdraw()
except:
    gui = False

def calculate_isi_violations(spike_times, spike_clusters, total_units, isi_threshold, min_isi):
    """
    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705
    spike_train : array of spike times
    min_time : minimum time for potential spikes
    max_time : maximum time for potential spikes
    isi_threshold : threshold for isi violation
    min_isi : threshold for duplicate spikes
    contam_perc : rate of contaminating spikes as a fraction of overall rate
        A perfect unit has a contam_perc = 0
        A unit with some contamination has a contam_perc < 0.5
        A unit with lots of contamination has a contam_perc > 1.0
    """
    cluster_ids = np.unique(spike_clusters)
    viol_rates = np.zeros((total_units,))
    for idx, cluster_id in enumerate(cluster_ids):
        for_this_cluster = (spike_clusters == cluster_id)
        
        spike_train = spike_times[for_this_cluster]
        min_time = np.min(spike_times[for_this_cluster])
        max_time = np.max(spike_times[for_this_cluster])
        
        duplicate_spikes = np.where(np.diff(spike_train) <= min_isi)[0]

        spike_train = np.delete(spike_train, duplicate_spikes + 1)
        isis = np.diff(spike_train)

        num_spikes = len(spike_train)
        num_violations = sum(isis < isi_threshold)
        violation_time = 2 * num_spikes * (isi_threshold - min_isi)
        total_rate = calc_FR(spike_train, min_time, max_time)
        violation_rate = num_violations/violation_time
        contam_perc = violation_rate/total_rate
        if contam_perc > 1:
            contam_perc = 1
        viol_rates[idx] = contam_perc

    return viol_rates

def calculate_firing_rate(spike_times, spike_clusters, total_units):
    cluster_ids = np.unique(spike_clusters)
    firing_rates = np.zeros((total_units,))
    min_time = np.min(spike_times)
    max_time = np.max(spike_times)
    for idx, cluster_id in enumerate(cluster_ids):
        for_this_cluster = (spike_clusters == cluster_id)
        firing_rates[idx] = calc_FR(spike_times[for_this_cluster],
                                        min_time = np.min(spike_times),
                                        max_time = np.max(spike_times))
    return firing_rates

def calculate_presence_ratio(spike_times, spike_clusters, total_units):
    cluster_ids = np.unique(spike_clusters)
    ratios = np.zeros((total_units,))
    for idx, cluster_id in enumerate(cluster_ids):
        for_this_cluster = (spike_clusters == cluster_id)
        ratios[idx] = calc_presense_ratio(spike_times[for_this_cluster],
                                                       min_time = np.min(spike_times),
                                                       max_time = np.max(spike_times))
    return ratios

def calculate_amplitude_cutoff(spike_clusters, amplitudes, total_units):
    cluster_ids = np.unique(spike_clusters)
    amplitude_cutoffs = np.zeros((total_units,))
    for idx, cluster_id in enumerate(cluster_ids):
        for_this_cluster = (spike_clusters == cluster_id)
        amplitude_cutoffs[idx] = calc_amp_cutoff(amplitudes[for_this_cluster])
    return amplitude_cutoffs    

def calc_FR(spike_train, min_time = None, max_time = None):
    """Calculate firing rate for a spike train.
    If no temporal bounds are specified, the first and last spike time are used.
    Inputs:
    -------
    spike_train : numpy.ndarray
        Array of spike times in seconds
    min_time : float
        Time of first possible spike (optional)
    max_time : float
        Time of last possible spike (optional)
    Outputs:
    --------
    fr : float
        Firing rate in Hz
    """
    if min_time is not None and max_time is not None:
        duration = max_time - min_time
    else:
        duration = np.max(spike_train) - np.min(spike_train)
    fr = spike_train.size / duration
    return fr

def calc_presense_ratio(spike_train, min_time, max_time, num_bins=100):
    """Calculate fraction of time the unit is present within an epoch.
    Inputs:
    -------
    spike_train : array of spike times
    min_time : minimum time for potential spikes
    max_time : maximum time for potential spikes
    Outputs:
    --------
    presence_ratio : fraction of time bins in which this unit is spiking
    """
    h, b = np.histogram(spike_train, np.linspace(min_time, max_time, num_bins))
    return np.sum(h > 0) / num_bins


def calc_amp_cutoff(amplitudes, num_histogram_bins = 500, histogram_smoothing_value = 3):
    """ Calculate approximate fraction of spikes missing from a distribution of amplitudes
    Assumes the amplitude histogram is symmetric (not valid in the presence of drift)
    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705
    Input:
    ------
    amplitudes : numpy.ndarray
        Array of amplitudes (don't need to be in physical units)
    Output:
    -------
    fraction_missing : float
        Fraction of missing spikes (0-0.5)
        If more than 50% of spikes are missing, an accurate estimate isn't possible
    """
    h,b = np.histogram(amplitudes, num_histogram_bins, density=True)
    pdf = gaussian_filter1d(h,histogram_smoothing_value)
    support = b[:-1]
    peak_index = np.argmax(pdf)
    G = np.argmin(np.abs(pdf[peak_index:] - pdf[0])) + peak_index
    bin_size = np.mean(np.diff(support))
    fraction_missing = np.sum(pdf[G:])*bin_size
    fraction_missing = np.min([fraction_missing, 0.5])
    return fraction_missing

params = {}
params['isi_threshold'] = 0.002
params['min_isi'] = 0.0005
params['isi_viol_th'] = 0.4 # 40% violations
params['presence_ratio'] = 0.5
params['firing_rate_th'] = 0.1 # 0.1Hz
params['amp_cutoff_th'] = 0.01
params['amp_th'] = 25 # 25uV 

fs = 30000.0 # sample at 30kHz

# get directories to process
dirs = []
if gui:
    print("\nA dialog box should appear to choose data directories. Press 'cancel' or ESC to end.")
    t = "Choose directory(s) to apply quality metrics to."
    while True:
        d = filedialog.askdirectory(mustexist=True, title=t)
        # windows ('nt') vs linux
        if os.name == 'nt':
            gui_root.attributes('-topmost', True, '-alpha', 0)
        if d == () or d == '':
            break
        else:
            dirs.append(d)
else: # e.g. running the script from the command line
    print()
    dirs_txt = "Which directories do you want to process? You may specify multiple. " 
    dirs_txt += "List your directories separated by commas. \n"
    dirs_txt += "E.g. C:\\animal1_day1, C:\\animal2_day6 \n\n"
    dirs = input(dirs_txt).replace(", ", ",").split(',')
    # in case someone is silly enough to append a trailing comma
    if dirs[-1] == "":
        del dirs[-1]
dirs = list(set(dirs)) # remove duplicates
assert len(dirs) > 0, "No input directories specified :("

# quick check that the files we'll need are present in each dir
for dirname in dirs:
    fname = os.path.join(dirname, 'spike_times.npy')
    assert os.path.isfile(fname), f'{fname} is missing'
    fname = os.path.join(dirname, 'cluster_info.tsv')
    err_txt = f'{fname} is missing. '
    err_txt += 'Open the data in phy and hit the save icon to generate it.'
    assert os.path.isfile(fname), err_txt

for dirname in dirs:
    spike_times = np.ravel(np.load(os.path.join(dirname,'spike_times.npy'), allow_pickle=True)) / fs
    spike_clusters = np.ravel(np.load(os.path.join(dirname,'spike_clusters.npy'), allow_pickle=True))
    # amplitudes = np.ravel(np.load(os.path.join(dirname,'amplitudes.npy'), allow_pickle=True))
    channel_map = np.load(os.path.join(dirname,'channel_map.npy'))[0]
    cluster_info = pd.read_csv(os.path.join(dirname,'cluster_info.tsv'), sep='\t')
    total_units = len(np.unique(spike_clusters))
    # filter out spikes that have bad values 
    spike_clusters = [c for t, c in zip(spike_times, spike_clusters) if not math.isnan(t) and not math.isinf(t)]
    spike_times = [t for t in spike_times if not math.isnan(t) and not math.isinf(t)]
    spike_clusters = np.array(spike_clusters)
    spike_times = np.array(spike_times)
    isi_viol = calculate_isi_violations(spike_times, spike_clusters, total_units, params['isi_threshold'], params['min_isi'])
    presence_ratio = calculate_presence_ratio(spike_times, spike_clusters, total_units)
    firing_rate = calculate_firing_rate(spike_times, spike_clusters, total_units)
    # amplitude_cutoff = calculate_amplitude_cutoff(spike_clusters, amplitudes, total_units)
    cluster_ids = np.unique(spike_clusters)
    
    metrics = pd.DataFrame(data= dict((
        ('cluster_id', cluster_ids),
        ('firing_rate' , firing_rate),
        ('presence_ratio' , presence_ratio),
        ('isi_viol' , isi_viol),
        # ('amp_cutoff' , amplitude_cutoff),
    )))
    metrics['group'] = cluster_info['group']
    metrics['depth'] = cluster_info['depth']
    metrics['ch'] = cluster_info['ch']
    metrics['num_spikes'] = cluster_info['n_spikes']
    
    # whether the cells pass all of our metrics / standards
    goodness = metrics['isi_viol'] < params['isi_viol_th']
    goodness = goodness & (metrics['presence_ratio'] > params['presence_ratio'])
    goodness = goodness & (metrics['firing_rate'] > params['firing_rate_th'])
    goodness = goodness & (metrics['group'] == 'good')
    metrics['good'] = goodness
    
    #save the metrics
    fname = os.path.join(dirname, 'UnitMetrics.csv')
    metrics.to_csv(fname, index=True)
    
    print()
    print('Number of unit metrics')
    print(np.nansum(metrics['good']))
    
    # save our spike_times and spike_clusters
    good_IDs = np.array(metrics['cluster_id'][metrics['good']])
    _spike_times = []
    _spike_clusters = []
    for id in good_IDs:
        # Get the only array in the tuple with '[0]'
        c_spike_times = spike_times[np.where(spike_clusters == id)[0]]
        _spike_times.append(c_spike_times)
        _spike_clusters.append(id)
    # tinyurl.com/4sw3bau9
    _spike_times = np.array(_spike_times, dtype='object')
    _spike_clusters = np.array(_spike_clusters)

    fname = os.path.join(dirname, 'post_ks_spike_times.npy')
    np.save(fname, _spike_times)
    fname = os.path.join(dirname, 'post_ks_spike_clusters.npy')
    np.save(fname, _spike_clusters)

if gui:
    gui_root.destroy()