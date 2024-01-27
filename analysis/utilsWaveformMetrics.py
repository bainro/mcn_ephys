# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 03:33:57 2023

@author: saxen
"""
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.signal import resample


def calculate_snr(W):
    
    """
    Calculate SNR of spike waveforms.

    Converted from Matlab by Xiaoxuan Jia

    ref: (Nordhausen et al., 1996; Suner et al., 2005)

    Input:
    -------
    W : array of N waveforms (N x samples)

    Output:
    snr : signal-to-noise ratio for unit (scalar)

    """

    W_bar = np.nanmean(W, axis=0)
    A = np.max(W_bar) - np.min(W_bar)
    e = W - np.tile(W_bar, (np.shape(W)[0], 1))
    snr = A/(2*np.nanstd(e.flatten()))

    return snr


def calculate_waveform_duration(waveform, timestamps):
    
    """ 
    Duration (in seconds) between peak and trough

    Inputs:
    ------
    waveform : numpy.ndarray (N samples)
    timestamps : numpy.ndarray (N samples)

    Outputs:
    --------
    duration : waveform duration in milliseconds

    """

    trough_idx = np.argmin(waveform)
    peak_idx = np.argmax(waveform)

    # to avoid detecting peak before trough
    if waveform[peak_idx] > np.abs(waveform[trough_idx]):
        duration =  timestamps[peak_idx:][np.where(waveform[peak_idx:]==np.min(waveform[peak_idx:]))[0][0]] - timestamps[peak_idx] 
    else:
        duration =  timestamps[trough_idx:][np.where(waveform[trough_idx:]==np.max(waveform[trough_idx:]))[0][0]] - timestamps[trough_idx] 

    return duration * 1e3


def calculate_waveform_halfwidth(waveform, timestamps):
    """ 
    Spike width (in seconds) at half max amplitude

    Inputs:
    ------
    waveform : numpy.ndarray (N samples)
    timestamps : numpy.ndarray (N samples)

    Outputs:
    --------
    halfwidth : waveform halfwidth in milliseconds

    """

    trough_idx = np.argmin(waveform)
    peak_idx = np.argmax(waveform)

    try:
        if waveform[peak_idx] > np.abs(waveform[trough_idx]):
            threshold = waveform[peak_idx] * 0.5
            thresh_crossing_1 = np.min(
                np.where(waveform[:peak_idx] > threshold)[0])
            thresh_crossing_2 = np.min(
                np.where(waveform[peak_idx:] < threshold)[0]) + peak_idx
        else:
            threshold = waveform[trough_idx] * 0.5
            thresh_crossing_1 = np.min(
                np.where(waveform[:trough_idx] < threshold)[0])
            thresh_crossing_2 = np.min(
                np.where(waveform[trough_idx:] > threshold)[0]) + trough_idx

        halfwidth = (timestamps[thresh_crossing_2] - timestamps[thresh_crossing_1]) 

    except ValueError:
        halfwidth = np.nan

    return halfwidth * 1e3


def calculate_waveform_PT_ratio(waveform):
    """ 
    Peak-to-trough ratio of 1D waveform

    Inputs:
    ------
    waveform : numpy.ndarray (N samples)

    Outputs:
    --------
    PT_ratio : waveform peak-to-trough ratio

    """

    trough_idx = np.argmin(waveform)
    peak_idx = np.argmax(waveform)
    PT_ratio = np.abs(waveform[peak_idx] / waveform[trough_idx])
    
    return PT_ratio


def calculate_waveform_repolarization_slope(waveform, timestamps, window=20):
    """ 
    Spike repolarization slope (after maximum deflection point)

    Inputs:
    ------
    waveform : numpy.ndarray (N samples)
    timestamps : numpy.ndarray (N samples)
    window : int
        Window (in samples) for linear regression

    Outputs:
    --------
    repolarization_slope : slope of return to baseline (V / s)

    """

    max_point = np.argmax(np.abs(waveform))
    waveform = - waveform * (np.sign(waveform[max_point])) # invert if we're using the peak
    repolarization_slope = linregress(timestamps[max_point:max_point+window], waveform[max_point:max_point+window])[0]

    return repolarization_slope * 1e-6



def calculate_waveform_recovery_slope(waveform, timestamps, window=20):
    """ 
    Spike recovery slope (after repolarization)

    Inputs:
    ------
    waveform : numpy.ndarray (N samples)
    timestamps : numpy.ndarray (N samples)
    window : int
        Window (in samples) for linear regression

    Outputs:
    --------
    recovery_slope : slope of recovery period (V / s)

    """
    max_point = np.argmax(np.abs(waveform))
    waveform = - waveform * (np.sign(waveform[max_point])) # invert if we're using the peak
    peak_idx = np.argmax(waveform[max_point:]) + max_point
    recovery_slope = linregress(timestamps[peak_idx:peak_idx+window], waveform[peak_idx:peak_idx+window])[0]
    
    return recovery_slope * 1e-6

def load(folder, filename):
    return np.load(os.path.join(folder, filename))

def read_cluster_group_tsv(filename):
    info = pd.read_csv(filename, sep='\t')
    cluster_ids = info['cluster_id'].values.astype('int')
    cluster_quality = info['group'].values
    return cluster_ids, cluster_quality

def load_kilosort_data(folder,  sample_rate = None, convert_to_seconds = True, template_zero_padding= 21):
    ampltiude = np.ravel(load(folder,'amplitudes.npy'))
    spike_times = load(folder,'spike_times.npy')
    spike_clusters = load(folder,'spike_clusters.npy')
    spike_templates = load(folder, 'spike_templates.npy')
    templates = load(folder,'templates.npy')
    unwhitening_mat = load(folder,'whitening_mat_inv.npy')
    channel_map = np.squeeze(load(folder, 'channel_map.npy'))
    cluster_ids, cluster_quality = read_cluster_group_tsv(os.path.join(folder, 'cluster_group.tsv'))
    cluster_info = pd.read_csv(os.path.join(folder, 'cluster_info.tsv'), delimiter='\t')
    
    templates = templates[:,template_zero_padding:,:] # remove zeros
    spike_clusters = np.squeeze(spike_clusters) # fix dimensions
    spike_templates = np.squeeze(spike_templates) # fix dimensions
    spike_times = np.squeeze(spike_times) # fix dimensions

    if convert_to_seconds and sample_rate is not None:
       spike_times = spike_times / sample_rate 
       
    unwhitened_temps = np.zeros((templates.shape))
    for temp_idx in range(templates.shape[0]):
        unwhitened_temps[temp_idx,:,:] = np.dot(np.ascontiguousarray(templates[temp_idx,:,:]),np.ascontiguousarray(unwhitening_mat))
                    
    return ampltiude, spike_times, spike_clusters, unwhitened_temps, channel_map, cluster_ids, cluster_info



# calculate waveform metrics
def calculate_waveform_metrics(waveforms, 
                               cluster_id, 
                               peak_channel, 
                               channel_map, 
                               sample_rate, 
                               upsampling_factor, 
                               spread_threshold,
                               site_spacing):    
    """
    Calculate metrics for an array of waveforms.

    Metrics come from Jia et al. (2019) High-density extracellular probes reveal 
    dendritic backpropagation and facilitate neuron classification. J Neurophys

    https://doi.org/10.1152/jn.00680.2018


    Inputs:
    -------
    waveforms : numpy.ndarray (num_spikes x num_channels x num_samples)
        Can include NaN values for missing spikes
    cluster_id : int
        ID for cluster
    peak_channel : int
        Location of waveform peak
    channel_map : numpy.ndarray
        Channels used for spike sorting
    sample_rate : float
        Sample rate in Hz
    upsampling_factor : float
        Relative rate at which to upsample the spike waveform
    spread_threshold : float
        Threshold for computing spread of 2D waveform
    site_range : float
        Number of sites to use for 2D waveform metrics
    site_spacing : float
        Average vertical distance between sites (m)
    epoch_name : str
        Name of epoch for which these waveforms originated

    Outputs:
    -------
    metrics : pandas.DataFrame
        Single-row table containing all metrics

    """

    mean_2D_waveform = np.squeeze(np.nanmean(waveforms[:, channel_map, :], 0))
    local_peak = np.argmin(np.abs(channel_map - peak_channel))

    num_samples = waveforms.shape[2]
    new_sample_count = int(num_samples * upsampling_factor)

    mean_1D_waveform = resample(mean_2D_waveform[local_peak, :], new_sample_count)

    timestamps = np.linspace(0, num_samples / sample_rate, new_sample_count)

    duration = calculate_waveform_duration(mean_1D_waveform, timestamps)
    halfwidth = calculate_waveform_halfwidth(mean_1D_waveform, timestamps)
    PT_ratio = calculate_waveform_PT_ratio(mean_1D_waveform)
    repolarization_slope = calculate_waveform_repolarization_slope( mean_1D_waveform, timestamps)
    recovery_slope = calculate_waveform_recovery_slope(mean_1D_waveform, timestamps)

    data = [[cluster_id, duration, halfwidth, PT_ratio, repolarization_slope, recovery_slope]]
    metrics = pd.DataFrame(data,
                           columns=['cluster_id', 'duration', 'halfwidth',
                                     'PT_ratio', 'repolarization_slope', 'recovery_slope'])

    return metrics

def extract_waveforms(raw_data, 
                      spike_times, 
                      spike_clusters, 
                      templates, 
                      cluster_ids,
                      peak_channels,
                      channel_map, 
                      bit_volts, 
                      sample_rate, 
                      site_spacing, 
                      params):
    
    """
    Calculate mean waveforms for sorted units.

    Inputs:
    -------
    raw_data : continuous data as numpy array (samples x channels)
    spike_times : spike times (in samples)
    spike_clusters : cluster IDs for each spike time []
    clusterIDs : all unique cluster ids
    cluster_quality : 'noise' or 'good'
    sample_rate : Hz
    site_spacing : m

    Parameters:
    ----------
    samples_per_spike : number of samples in extracted spikes
    pre_samples : number of samples prior to peak
    num_epochs : number of epochs to calculate mean waveforms
    spikes_per_epoch : max number of spikes to generate average for epoch

    """

    # #############################################

    samples_per_spike = params['samples_per_spike']
    pre_samples = params['pre_samples']
    spikes_per_epoch = params['spikes_per_epoch']
    upsampling_factor = params['upsampling_factor']
    spread_threshold = params['spread_threshold']

    # #############################################

    metrics = pd.DataFrame()

    total_units = len(cluster_ids)

    mean_waveforms = np.zeros((total_units, 2, raw_data.shape[1], samples_per_spike))

    for cluster_idx, cluster_id in enumerate(cluster_ids):
        in_cluster = (spike_clusters == cluster_id)

        if np.sum(in_cluster) > 0:
            times_for_cluster = spike_times[in_cluster]

            waveforms = np.empty((spikes_per_epoch, raw_data.shape[1], samples_per_spike))
            waveforms[:] = np.nan
            np.random.shuffle(times_for_cluster)

            total_waveforms = np.min([times_for_cluster.size, spikes_per_epoch])

            for wv_idx, peak_time in enumerate(times_for_cluster[:total_waveforms]):
                start = int(peak_time-pre_samples)
                end = start + samples_per_spike
                rawWaveform = raw_data[start:end, :].T

                # in case spike was at start or end of dataset
                if rawWaveform.shape[1] == samples_per_spike:
                    waveforms[wv_idx, :, :] = rawWaveform * bit_volts

            # concatenate to existing dataframe
            metrics = pd.concat([metrics, calculate_waveform_metrics(waveforms[:total_waveforms, :, :],
                                                                     cluster_id, 
                                                                     peak_channels[cluster_idx], 
                                                                     channel_map,
                                                                     sample_rate, 
                                                                     upsampling_factor,
                                                                     spread_threshold,
                                                                     site_spacing,
                                                                     )])
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_waveforms[cluster_idx, 0, :, :] = np.nanmean(waveforms, 0)
                mean_waveforms[cluster_idx, 1, :, :] = np.nanstd(waveforms, 0)

                # remove offset
                for channel in range(0, mean_waveforms.shape[3]):
                    mean_waveforms[cluster_idx, 0, channel, :] = mean_waveforms[cluster_idx, 0, channel, :] - mean_waveforms[cluster_idx, 0, channel, 0]
    
    metrics.reset_index()
    
    # calculate ampltiude for each cluster
    amplitude = []
    for mwf, pc in zip(mean_waveforms, peak_channels):
        mwf = mwf[0, pc,:]
        max_amp = np.max([np.max(mwf), np.abs(np.min(mwf))])
        if np.abs(np.min(mwf))>np.max(mwf):
            max_amp = -max_amp
        baseline = np.nanmedian(mwf[:5])
        amplitude.append(np.abs(max_amp - baseline))
    metrics['amp'] = amplitude   

    return mean_waveforms, metrics