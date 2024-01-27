# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:48:43 2023

@author: jshobe
"""
import numpy as np
import pandas as pd
import scipy.io as spio
import scipy.stats as spst
import matplotlib.pyplot as plt

# load intan timestamps
def loadintants(filename, fs = 30000.0, stidx=None, etidx=None):
    dig_data = np.array(np.load(filename, mmap_mode='r'), 'float32')
    # create timestamp matrix   
    numSamples = len(dig_data)
    dt = 1./fs
    ts = np.arange(0,numSamples*dt,dt)
    # load the digital datapoints
    idx = np.where(dig_data<2)[0]
    dig_data[idx] = dig_data[idx] + 2
    # **************************************************find 1st transition point
    diff_dig_data = np.ediff1d(dig_data) 
    # find all the transition datasets
    movement_ind1 = np.where(diff_dig_data==1)[0]
    movement_ind2 = np.where(diff_dig_data==-1)[0]
    movement_ind = np.sort(np.concatenate((movement_ind1, movement_ind2)))
    # movement change timestamps
    intan_aligned_ts = ts[movement_ind]
    if stidx is not None:
        intan_aligned_ts = intan_aligned_ts[stidx:]
    if etidx is not None:
        intan_aligned_ts = intan_aligned_ts[:etidx]
    del dig_data
    return intan_aligned_ts


# normalize 1d array bw 0 and 1
def norm1D(arr):
    return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))


# read behavior file
def readBehavFile(dirname):
    # load the raw data
    garr_data = spio.loadmat(dirname)
    garr = garr_data['garr']
    garr = garr[:-1,:]
    return garr

    
# process behavior data 
def processBehavFile(garr, samplerate=30, VRradius=50, intants=None):
    # position in laps and in cm across the circumference
    VRcircumference = 2*np.pi*VRradius
    lappos = garr[:,4] 
    poscm = (lappos%1)*VRradius*2.*np.pi
    pos, _ = np.modf(lappos)
    # instant velocity
    vel = np.diff(lappos)  #laps/sample
    vel = np.insert(vel,0,0)
    vel = vel*VRcircumference*samplerate #cm/s
    samples = np.arange(len(lappos))
    # timestamp in seconds
    time = samples/samplerate 
    lapnum = np.ones(len(lappos))
    # reward and licks
    reward = np.array(np.diff(garr[:,2])<0, int)
    reward = np.insert(reward,0,0)
    # add lap number 
    hallChangeIdx = np.where(np.diff(pos)<-0.5)[0]
    HallChangeStartidx = np.insert(hallChangeIdx+2,0,1)
    HallChangeEndidx = np.append(hallChangeIdx+2, len(lappos))
    lapn = 1
    for st,et in zip(HallChangeStartidx,HallChangeEndidx):
        lapnum[st:et+1] = lapn
        lapn+=1
    # create dataframe with all variables
    garrN = pd.DataFrame(garr, columns=['rev','vel','rew','lick','position','obj','hallnum','systime'])
    garrN['vel'] = np.array(vel, dtype=np.float32)
    garrN['rew'] = np.array(reward, dtype=np.float32)
    garrN['lapnum'] = lapnum
    garrN['pos'] = pos
    garrN['poscm'] = poscm
    garrN['time'] = time
    # routine to align intan and pos change file
    # we need movement data only, no stops
    posChange = np.ediff1d(garrN['pos'])
    posChangeDiffIdx = np.where(posChange!=0)[0]
    garrN = garrN.iloc[posChangeDiffIdx]
    garrN = garrN.reset_index(drop=True)
    garrN = garrN.drop(columns=['rev', 'lick', 'systime','position'])
    if intants is not None:
        garrN['time'] = intants
        vel = np.diff(garrN['poscm'])/np.diff(intants)
        vel = np.insert(vel,0,np.nan)
        garrN['vel'] = vel
    return garrN


# find time in single trials
def time_in_single_trial(value):
    if len(value)>1:
        idx = np.where(value!=np.nan)[0]
        if len(idx)>=2:
            value = value.iloc[idx]
            return value.iloc[-1] - value.iloc[0]
        else:
            return np.nan
    else:
        return np.nan
    
    
# get binned trial data
def calcBinnedData(trial_movement, binwidth=1):
    bins = np.arange(0,1,binwidth)
    # matrix to hold the binned speed trials, binned time count, binned time sum
    binned_speed = np.empty((len(trial_movement), len(bins)))
    binned_speed[:] = np.nan
    binned_timecount = np.empty((len(trial_movement), len(bins))) 
    binned_timecount[:] = np.nan
    binned_timesum = np.empty((len(trial_movement), len(bins)))
    binned_timesum[:] = np.nan
    # iterate over each trial data
    for tnum in range(len(trial_movement)):
        # load individual trial data
        mvmtdat = trial_movement[tnum]
        # assign bins for pos
        inds = np.digitize(mvmtdat['pos'], bins)
        # find unique bin array
        inds_unique = np.unique(inds)-1

        # temp dataframe to ease calculation
        tempdf = pd.DataFrame({'inds':inds, 'vel':mvmtdat['vel'], 'pos':mvmtdat['pos'], 'time':mvmtdat['time']})
        
        # binned speed calculation
        tempspd = tempdf.groupby('inds')['vel'].agg(np.nanmean)
        # create empty binned speed array for each trial
        # assign mean grouped speed to their respective indices
        binned_speed_ = np.zeros(len(bins))
        binned_speed_[inds_unique] = tempspd
        binned_speed[tnum,:] = binned_speed_
        
        # binned time count calculation
        temptimec = tempdf.groupby('inds')['pos'].agg('count')
        # create empty binned time count array for each trial
        # assign time count to their respective indices
        binned_tc_ = np.zeros(len(bins))
        binned_tc_[inds_unique] = temptimec
        binned_timecount[tnum,:] = binned_tc_
        
        # binned time sum calculation
        time_spent = tempdf.groupby('inds')['time']
        time_spent = time_spent.apply(time_in_single_trial)
        # assign count of time to their respective indices
        binned_timesum_ = np.zeros(len(bins))
        binned_timesum_[inds_unique] = time_spent
        binned_timesum[tnum,:] = binned_timesum_
        
        idx  = binned_speed>120
        binned_speed[idx] = np.nan
        binned_timecount[idx] = np.nan
        binned_timesum[idx] = np.nan
    return binned_speed, binned_timecount, binned_timesum, bins


# find trial transition timestamps
def trialTransition(df):
    # HACKY WAY, but it works
    trial_endidx = np.where(np.diff(df['pos'])<-0.95)[0]+1
    trial_startidx = np.where(np.diff(df['pos'])<-0.95)[0]+1
    trial_startidx = np.insert(trial_startidx,0,0)
    trial_endidx = np.append(trial_endidx, len(df)-1)
    return trial_startidx, trial_endidx


# calculate trial data for each hallway
def calcTrialData(dfhnum, trial_start, trial_end):
    # variables to hold the trial movement and trial lick dict
    trial_movement = {}
    # iterate over each trial
    for tnum, (st,et) in enumerate(zip(trial_start,trial_end)):
        # get speed, position, time  for each trial
        pos = np.array(dfhnum[st:et]['pos'])
        time = np.array(dfhnum[st:et]['time'])
        vel = np.array(dfhnum[st:et]['vel'])
        # add all the data to trial movement 
        trial_movement[tnum] = {'vel':vel, 'pos':pos, 'time':time}
    return trial_movement


# calculate hallways specific data
def calcHallwayData(df, hnum, binwidth=1):
    # create hallway specific dataframe
    idx = np.where(np.array(df['hallnum'],dtype=int)==hnum)[0]
    df_hnum = df.loc[idx,:].reset_index(drop=True)
    # remove buggy matlab hallway number
    dropidx = np.where(df_hnum['lapnum'].value_counts()<=2)[0]
    df_hnum = df_hnum.drop(dropidx).reset_index(drop=True)
    # find trial start and end indices
    trial_start_idx, trial_end_idx = trialTransition(df_hnum)
    trial_transition = np.vstack((trial_start_idx, trial_end_idx)).T
    diffidx = trial_transition[:,1] - trial_transition[:,0]
    trial_transition = np.delete(trial_transition, np.where(diffidx<5)[0], axis=0)
    trial_start_idx, trial_end_idx = trial_transition[:,0], trial_transition[:,1]
    trial_transition = np.array(df_hnum['time'])[trial_transition]
    # calculate trial specific data
    trial_data = calcTrialData(df_hnum, trial_start_idx, trial_end_idx)
    # get binned speed, binned time count, binned time sum plot
    bintrial_speed, bintrial_timec, bintrial_timesum, bins = calcBinnedData(trial_data, binwidth=binwidth)
    
    # create the dictionary to late store the data
    return {'trial_data':trial_data, 
            'binned_speed':bintrial_speed, 
            'binned_time':bintrial_timec,
            'binned_time_sum':bintrial_timesum,
            'trial_transition':trial_transition,
            'xbins':bins}

# plot all hallways data
def plotSpeedAcrossHallways(allhallwaydata, figname, aname, savefig=False, bin2cm=3.0):
    plt.figure(figsize=(11,5))
    colors = plt.cm.jet(np.linspace(0,1,len(allhallwaydata)))
    for c,hnum in enumerate(list(allhallwaydata.keys())):
        xdat = allhallwaydata[hnum]['xbins']
        ydat = np.nanmean(allhallwaydata[hnum]['binned_speed'],0)
        yerr = spst.sem(allhallwaydata[hnum]['binned_speed'],0, nan_policy='omit')
        # plt.plot(xdat, ydat, label=ENVNUM[str(hnum)], color=colors[c])
        plt.plot(xdat, ydat, label=str(hnum), color=colors[c])
        plt.fill_between(xdat, ydat-yerr, ydat+yerr, color=colors[c], alpha=0.4)
    plt.xticks(xdat[::20], np.round(xdat[::20]*bin2cm*100,1))
    plt.legend(loc='lower left')
    plt.xlabel('Position (cm)')
    plt.ylabel('Speed (cm/s)')
    plt.title(aname)
    plt.tight_layout()
    if savefig:
        plt.savefig(figname, dpi=300)
        plt.close()
    return None