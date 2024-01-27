# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 19:24:33 2023

@author: saxena
"""
import os
import warnings
import numpy as np
import pandas as pd
import utilsClusterQual as ucq
import utilsWaveformMetrics as uwm
import matplotlib.pyplot as plt
warnings.simplefilter('ignore')

# other init params
vertical_site_spacing = 25e-6 # in meters
bit_volts = 0.195
fs = 30000.0

# parameters for waveform metrics calculation
params = {}
params['samples_per_spike'] = 82
params['pre_samples'] = 20
params['spikes_per_epoch'] = 250
params['upsampling_factor'] = 200/82
params['spread_threshold'] = 0.12

# parameters for cluster quality metrics calculation
qualParams = {} 
qualParams['isi_threshold'] = 0.002 #refractory period
qualParams['min_isi'] = 0.0002
qualParams['isi_viol_th'] = 0.2 # 20% contamination
qualParams['presence_ratio'] = 0.7 # firing in 70% of behavior epochs
qualParams['firing_rate_th'] = 0.01  # firing rate above 0.01 Hz
#qualParams['amp_cutoff_th'] = 0.2 # missing only 20% of spikes below threshold
qualParams['amp_th'] = 30 # ampltiude greater than 30 uV 

# read csv with start and end time for each experimental animal
dirname = r'.\Spikesorted-SWIL'
epochsfname = 'swil-animals.csv'
epochsdf = pd.read_csv(os.path.join(dirname,epochsfname))
filename = epochsdf['file_name']
start_time, end_time = epochsdf['start_time'], epochsdf['end_time']

# loop through all directories
for dname, st, et in zip(filename, start_time, end_time):
    print('.........')
    print('Processing ' + str(dname))
    aname = dname
    dname = os.path.join(dirname,dname)
    
    # load kilosort data and convert the channel position to microns
    channel_positions = np.load(os.path.join(dname,'channel_positions.npy'))
    channel_positionsX = channel_positions[:,0]*0.001
    channel_positionsY = channel_positions[:,1]
    amplitudes, spike_times, spike_clusters, templates, channel_map, cluster_ids, cluster_info = \
        uwm.load_kilosort_data(dname, fs, convert_to_seconds = False)
    cluster_info = cluster_info[cluster_info['group']=='good']
    print('Finished loading kilosort data..')

    # load raw dat file data to load waveforms
    num_channels = len(channel_map)
    if os.path.isfile(os.path.join(dname, 'temp_wh.dat')):
        rawData = np.memmap(os.path.join(dname, 'temp_wh.dat'), dtype='int16', mode='r')
    else:
        bin_file = [file for file in os.listdir(dname) if file.endswith('.bin')]
        bin_file = bin_file[0]
        rawData = np.memmap(os.path.join(dname, bin_file), dtype='int16', mode='r')
    data = np.reshape(rawData, (int(rawData.size/num_channels), num_channels))
    del rawData
    print('Finished loading raw binary file..')
    
    # extract waveforms metrics
    peakch = np.array(cluster_info['ch'])
    goodcluid = np.array(cluster_info['cluster_id'])
    waveforms, wfMetrics = uwm.extract_waveforms(data, spike_times, spike_clusters, templates, goodcluid, peakch, channel_map, bit_volts, fs, vertical_site_spacing, params)
    wfMetrics = wfMetrics.reset_index(drop=True)
    print('Finished loading waveform metrics..')
    del data
    
    # load cellexplorer metrics file
    if os.path.exists(os.path.join(dirname, 'analyzedMetrics', aname+'-CellExplorerUnitMetrics.csv')):
        ceMetrics = pd.read_csv(os.path.join(dirname, 'analyzedMetrics', aname+'-CellExplorerUnitMetrics.csv'))
        ceMetrics = ceMetrics.rename(columns={"cluID": "cluster_id"})
    print('Finished loading cell explorer metrics..')
    
    # load cluster quality metrics
    if 'SWIL15VC' in dname or 'SWIL12PPC' in dname or 'SWIL20PPC' in dname:
        cluMetrics, spiketimesGood, spikeclustersGood = ucq.calcQualityMetrics(amplitudes, spike_times, spike_clusters, cluster_info, wfamp=wfMetrics['amp'], ampCE=None, epoch=[st,et], params=qualParams) #
    else:
        cluMetrics, spiketimesGood, spikeclustersGood = ucq.calcQualityMetrics(amplitudes, spike_times, spike_clusters, cluster_info, wfamp=None, ampCE=ceMetrics['ampCE'], epoch=[st,et], params=qualParams) #
    print('Finished loading cluster quality metrics..')
    
    # concatenate data
    metrics = pd.merge(wfMetrics, cluMetrics, on="cluster_id", how="left")
    waveforms = waveforms[metrics['isGood'],0,:,:]
    metrics = metrics[metrics['group']=='good']
    metrics = pd.merge(metrics, ceMetrics, on="cluster_id", how="left")
    metrics = metrics[metrics['isGood']]
    del cluMetrics, wfMetrics, ceMetrics
    
    # store data
    print('Number of good clusters: ' + str(sum(metrics['isGood'])))
    metrics.to_csv(os.path.join(dirname, 'analyzedMetrics', aname+'-UnitMetrics.csv'))
    np.save(os.path.join(dname,'proc-waveforms.npy'), waveforms)
    np.save(os.path.join(dname,'proc-spiketimes.npy'), spiketimesGood)
    np.save(os.path.join(dname,'proc-spikeclusters.npy'), spikeclustersGood)
    print('Finished saving processed ' + str(dname))
    
    # get good cluster x and y position
    cluster_xpos = []
    cluster_ypos = []
    good_clusters_channel_index = []
    for cnum in metrics[metrics['isGood']]['ch']:
        good_clusters_channel_index.append((np.where(cnum==channel_map)[0][0]))
        cluster_xpos = np.array(channel_positionsX[good_clusters_channel_index])
        cluster_ypos = np.array(channel_positionsY[good_clusters_channel_index]*0.001)
    
    # plot data
    fig = plt.figure(figsize=(6,10))
    plt.scatter(channel_positionsX, channel_positionsY*0.001, c='gray', marker='s', alpha=0.25)
    plt.scatter(cluster_xpos+np.random.uniform(-0.01,0.01,len(cluster_xpos)), 
                cluster_ypos+np.random.uniform(-0.01,0.01,len(cluster_ypos)), 
                c='b', s=20, cmap='plasma')
    plt.xlabel('Channel X Position', fontsize=16)
    plt.ylabel('Deep -> Sup layers (mm)', fontsize=16)
    for pos in ['right','top']:
        plt.gca().spines[pos].set_visible(False)
    plt.title('N='+str(sum(metrics['isGood'])), fontsize=18)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(dname,'proc-probemap.png'), dpi=200)
    print('Finished plotting probe map data..')
    print('.........')