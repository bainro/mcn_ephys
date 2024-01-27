import os
import bisect, sys
import numpy as np
from itertools import repeat
import scipy.ndimage as scnd
import matplotlib.pyplot as plt
import skimage.measure as measure
import multiprocessing as mp
sys.path.append('../LFP')
import mea
from detect_peaks import detect_peaks

#Find rightmost value less than or equal to x
def find_le(a, x):
    i = bisect.bisect_right(a, x)
    if i:
        return i-1, a[i-1]

#Find leftmost item greater than or equal to x
def find_ge(a, x):
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return i, a[i]
    
# function to generate normalized 1d array
def norm1d(arr):
    return arr/np.nanmax(arr)

# function to generate 1d histogram
def hist_1d(a):
    return np.histogram(a, bins=np.arange(20,92,1))[0]

# function to run gaussian smoothing
def filter_1d(a):
    return scnd.filters.gaussian_filter(a, 1.0)
    
# binary search function
def binarySearch(data, val):
    lo, hi = 0, len(data) - 1
    best_ind = lo
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if data[mid] < val:
            lo = mid + 1
        elif data[mid] > val:
            hi = mid - 1
        else:
            best_ind = mid
            break
        # check if data[mid] is closer to val than data[best_ind] 
        if abs(data[mid] - val) < abs(data[best_ind] - val):
            best_ind = mid
    return best_ind, data[best_ind]

# load spike times and cluster id for good units
def loadGoodSpikeTimes(dirname, metricsdf):
    clusterId = np.array(metricsdf.cluster_id)
    if os.path.exists(os.path.join(dirname, 'proc-spiketimes.npy')) and os.path.exists(os.path.join(dirname, 'proc-spikeclusters.npy')):
        spikeclusters = np.load(os.path.join(dirname, 'proc-spikeclusters.npy'), allow_pickle=True)
        if (spikeclusters==clusterId).all():
            spiketimes = np.load(os.path.join(dirname, 'proc-spiketimes.npy'), allow_pickle=True)
            return spiketimes, clusterId
    return None

# function to load the trial by trial data ocupancy data
# posX - linear position
# posT - time for each position point
# posSpeed - speed for each position point
# trialStart - trial start timestamps
# trialEnd - trial end timestamps
def loadOccupancy(hallway_trial_data, posMin=0, posMax=314, binwidth=4, speedThr=4, dt=1./30):
    # variables to hold the position data
    posx = []
    post = []
    posSpd = []
    trialSt = []
    trialEt = []
    # occupancy map for each trials
    omaptrial = []
    # iterate thorugh each trial and interpolate the time 
    # values to ensure uniform sampling rate (fs=30 Hz)
    for x in hallway_trial_data.values():
        pos = x['pos']*posMax # position
        time = x['time'] # intan time
        if len(pos)>5:
            # interpolate so that sampling rate = 30hz
            new_time = np.arange(np.nanmin(time), np.nanmax(time)-dt, dt)
            new_pos = np.interp(new_time, time, pos)
            speed = np.ediff1d(new_pos)/np.ediff1d(new_time)
            speed = np.insert(speed,0,np.nan)
            if new_time[0]>=0:
                trialSt.append(np.round(new_time[0],2))
                trialEt.append(np.round(new_time[-1],2))
                # remove low and very high speed values
                rmv_idx = np.where((speed<=speedThr) | (speed>=120))[0]
                new_pos[rmv_idx] = np.nan
                # trial by trial occupancy map
                omap, omapbins = np.histogram(new_pos, np.arange(posMin,posMax+binwidth,binwidth))
                omaptrial.append(np.array(omap, dtype='float'))
                # note down start and end time of each trials 
                posx.extend(new_pos)
                post.extend(new_time)
                posSpd.extend(speed)
    posx = np.array(posx)
    post = np.array(post)
    posSpd = np.array(posSpd)
    trialSt = np.array(trialSt)
    trialEt = np.array(trialEt)
    omaptrial = np.array(omaptrial)
    return posx, post, posSpd, trialSt, trialEt, omaptrial

# function to process occupancy data
def processOccupancy(omaptrial, posx,  posMin=0, posMax=400, binwidth=4, smwindow=0.25):
    # smooth the occupancy map across trials
    omaptrialsm = np.apply_along_axis(filter_1d, 1, omaptrial)
    # normalized occupancy map across trials
    omaptrialnorm = []
    for omtn in omaptrial:
        omaptrialnorm.append(omtn/np.nanmax(omtn))
    omaptrialnorm = np.array(omaptrialnorm)
    # generate occupancy map
    omap1d, omapbins = np.histogram(posx, np.arange(posMin,posMax+binwidth,binwidth))
    omap1dsm = scnd.filters.gaussian_filter(omap1d, smwindow)
    # return the data
    return omaptrialsm, omaptrialnorm, omap1d, omap1dsm, omapbins

# function to process spike pos data
def processSpikePos(spikets, spikephasehc, spikephaseself, trialStart, trialEnd, posX, posT, posSpd, lfpts):
    # variables to hold all the lap wise position and spike data
    # as well as overall spike pos, ts
    spkpos = []
    spkposts = []
    spkposspeed = []
    trspkpos = []
    trspkposts = []
    spkphasehc = []
    spkphaseself = []
    # find spike time and pos for each trial
    for tss, tse in zip(trialStart, trialEnd):
        # variables to hold trial by trial spike pos, spike ts
        tspkpos = []
        tspkposts = []
        # ensure that the spiketimestamps are after the start and end of trials
        ind = np.where((spikets>=posT[0]) & (spikets<=posT[-1]) & (spikets>=tss) & (spikets<=tse))[0]
        spkts = spikets[ind]
        spkphshc = spikephasehc[ind]
        spkphsself = spikephaseself[ind]
        for ts, phshc, phsself in zip(spkts, spkphshc, spkphsself):
            if ts>=posT[0] and ts<=posT[-1]:
                ind, val = binarySearch(posT, ts)
                # add the timestamps to overall spike data per cell
                # and trial by trial spike data per cell
                spkpos.append(posX[ind])
                spkposts.append(val)
                spkposspeed.append(posSpd[ind])
                tspkpos.append(posX[ind])
                tspkposts.append(val)
                spkphasehc.append(phshc)
                spkphaseself.append(phsself)
        # add the individual trial data to overall trial by trial data
        trspkpos.append(tspkpos)
        trspkposts.append(tspkposts)
    spkpos = np.array(spkpos)
    spkposts = np.array(spkposts)
    spkposspeed = np.array(spkposspeed)
    spkphasehc = np.array(spkphasehc)
    spkphaseself = np.array(spkphaseself)
    trspkpos = np.array(trspkpos)
    trspkposts = np.array(trspkposts)
    # return spike data
    return spkpos, spkposts, spkposspeed, spkphasehc, spkphaseself, trspkpos, trspkposts

# fucntion process spikemap data
def processSpikeMap(spkpostrial, spkpos, omapbins, posMin=0, posMax=314, binwidth=4, smwindow=1.0):
    spkmaptrials = []
    for trspkpos in spkpostrial:
        spikecnt, _ = np.histogram(trspkpos, np.arange(posMin,posMax+binwidth,binwidth))
        spkmaptrials.append(spikecnt)
    spkmaptrials = np.array(spkmaptrials, dtype='float')
    spkmaptrials[np.isnan(spkmaptrials)] = 0.0
    # smoothed spike map across trials
    spkmaptrialsm = np.apply_along_axis(filter_1d, 1, spkmaptrials)
    spkmaptrialsm[np.isnan(spkmaptrialsm)] = 0.0
    # normalized spike map across trials
    spkmaptrialnorm = []
    for spktsm in spkmaptrials:
        spkmaptrialnorm.append(spktsm/np.nanmax(spktsm))
    spkmaptrialnorm = np.array(spkmaptrialnorm)
    # spikemap average over the session
    spkmap1d, _ = np.histogram(spkpos, bins=omapbins)
    spkmap1d = np.array(spkmap1d, dtype='float')
    # filter the spikemap average over the session
    spkmap1dsm = scnd.filters.gaussian_filter(spkmap1d, smwindow)
    return spkmaptrials, spkmaptrialsm, spkmaptrialnorm, spkmap1d, spkmap1dsm
    
# function to calculate ratemap
def processRateMap(spkmaptr, omaptr, spkmaptrsm, omaptrsm, spkmap1d, omap1d, spkmap1dsm, omap1dsm, fs=30.0):
    rmaptr = (spkmaptr/omaptr)*fs
    rmaptr[np.isinf(rmaptr)] = 0.0
    rmaptr[np.isnan(rmaptr)] = 0.0
    # smoothed ratemap across trials
    rmaptrsm = (spkmaptrsm/omaptrsm)*fs
    rmaptrsm[np.isinf(rmaptrsm)] = 0.0
    rmaptrsm[np.isnan(rmaptrsm)] = 0.0
    # normalized firing rate map across trials
    rmaptrnorm = []
    for rt in rmaptrsm:
        rmaptrnorm.append(rt/np.nanmax(rt))
    rmaptrnorm = np.array(rmaptrnorm)
    # raw ratemap for entire session
    rmap1d = (spkmap1d/omap1d)*fs
    # smooth ratemap for the entire session
    rmap1dsm = (spkmap1dsm/omap1dsm)*fs
    # normalized firing rate map
    rmapnorm = np.apply_along_axis(norm1d, axis=0, arr=rmap1dsm)                
    return rmaptr, rmaptrsm, rmaptrnorm, rmap1d, rmap1dsm, rmapnorm

# function to compute spatial information score
def calcSpatialInformationScore(rateMap, occmap):    
    si = 0
    occmap[np.isnan(occmap)] = 0.0
    if np.nanmax(rateMap)>=0:
        #firing rates in occupied pixels
        rocc = rateMap[occmap>=0]
        occ = occmap[occmap>=0]
        occ = occ/np.nansum(occ)
        #this threshold is used in Jim's program & oiginal Skaggs c program- you need to avoid pixels 
        #with rates = 0, since log(0) is undefined, but using 0.0001 as threshold instead of > 0 
        #doesn't really hurt the info score significantly
        mrate = np.mean(rocc)
        for r,p_i in zip(rocc, occ):
            if r> 0.0001:
                si = si + p_i*(r/mrate)*np.log2(r/mrate)
        if si<0:
            si=0
    else:
        si = np.nan
    return np.round(si,2)

# calclate shuffled spatial info
def calcShuffledSI(idx, observedSI, spikets, occmap1d, endTime, startTime, trialStart, trialEnd, posX, posT, posSpeed, occmapbins, posMin=0, posMax=314, binwidth=3.14, fs=30.):
    #get a lag from 15seconds to behavior duration - 15seconds
    lag = np.random.uniform(15, (endTime-startTime-15), 1)[0]
    #add the lag to generate timestamps
    newSpikeTimestamps = spikets + lag
    #remove spike timestamps which do not fall in maze time
    #find the indices where spiketimestamps exceed end maze time
    shuffledSpikeTimestamps = posT[0] + (newSpikeTimestamps - posT[0]) % (posT[-1] - posT[0])
    # function to process spike pos data
    spikepos, _, _, spikepostrial, _ = processSpikePos(shuffledSpikeTimestamps, trialStart, trialEnd, posX, posT, posSpeed)
    # generate spike map across trials
    _, _, _, _, spikemap1dsm = processSpikeMap(spikepostrial, spikepos, occmapbins, posMin=posMin, posMax=posMax, binwidth=binwidth)
    ratemap1dsm = (spikemap1dsm/occmap1d)*fs
    si = calcSpatialInformationScore(ratemap1dsm, occmap1d)
    return si

# calclate shuffled spatial info by shuffling for individual trials
def calcShuffledSIByTrial(idx, observedSI, spikets, occmap1d, endTime, startTime, trialStart, trialEnd, posX, posT, posSpeed, occmapbins, posMin=0, posMax=314, binwidth=3.14, fs=30.):
    spkmaptrials = []
    for ts,te in zip(trialStart, trialEnd):
        #get a lag from 15seconds to behavior duration - 15seconds
        lag = np.random.uniform(1, (te-ts), 1)[0]
        spk = spikets[(spikets>=ts) & (spikets<=te)]
        #remove spike timestamps which do not fall in maze time
        shuffledSpikeTimestamps = ts + (spk + lag - ts) % (te-ts)
        # get spike pos
        spkpos = []
        for sst in shuffledSpikeTimestamps:
            if sst>=ts and sst<=te:
                ind, val = binarySearch(posT, sst)
                spkpos.append(posX[ind])
        spkpos = np.array(spkpos)
        # function to process spike pos data
        spikecnt, _ = np.histogram(spkpos, np.arange(posMin,posMax+binwidth,binwidth))
        spkmaptrials.append(spikecnt)
    spkmaptrials = np.array(spkmaptrials, dtype='float')
    spkmaptrials[np.isnan(spkmaptrials)] = 0.0
    # smoothed spike map across trials
    spkmaptrialsm = np.apply_along_axis(filter_1d, 1, spkmaptrials)
    spkmaptrialsm[np.isnan(spkmaptrialsm)] = 0.0
    spikemap1dsm = np.nanmean(spkmaptrialsm,0)
    ratemap1dsm = (spikemap1dsm/occmap1d)*fs
    si = calcSpatialInformationScore(ratemap1dsm, occmap1d)
    return si

# function to get shuffled spatial information
def calcShuffleSpatialInfo(observedSI, spikets, occmap1d, endTime, startTime, trialStart, trialEnd, posX, posT, posSpeed,  occmapbins, posMin=0, posMax=314, binwidth=4, fs=30.):
    idx = np.arange(100)
    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool(10)
    shuffledsi = pool.starmap(calcShuffledSIByTrial, zip(idx, repeat(observedSI), repeat(spikets), repeat(occmap1d), 
                                             repeat(endTime), repeat(startTime), repeat(trialStart), 
                                             repeat(trialEnd), repeat(posX), repeat(posT), repeat(posSpeed), 
                                             repeat(occmapbins))) 
    pool.close() 
    pool.join()
    return 1 - np.nansum(observedSI>shuffledsi)/len(shuffledsi), np.array(shuffledsi)

# get sparsity calculation
def calcSparsity(rmap, rmaptrials):
    sparsity = np.nanmean(rmap)**2/ np.nanmean(rmap**2)
    sparsityTrials = []
    for rt in rmaptrials:
        sp = np.nanmean(rt)**2/ np.nanmean(rt**2)
        sparsityTrials.append(sp)
    return np.round(sparsity,3), np.round(np.array(sparsityTrials),3)
    
# get field stability
def calcStabilityIndex(rmaptrials):
    corr_ = np.corrcoef(rmaptrials)
    corr_ = np.triu(corr_)
    np.fill_diagonal(corr_,np.nan)
    corr_[np.tril_indices(corr_.shape[0], -1)] = np.nan
    stabilityInd = np.nanmean(corr_)
    return np.round(stabilityInd,4)

# rotate place map
def rotatedPlacemap(rmap, rmaptrial):
    valleyind = detect_peaks(rmap, mph=-2, valley=True)
    if len(valleyind):
        if len(valleyind)>=2:
            valleyind = valleyind[1]
        else:
            valleyind = valleyind[0]
        rmap = np.roll(rmap,-valleyind)
        rmaptrial = np.roll(rmaptrial, -valleyind, axis=1) 
    return rmap, rmaptrial, valleyind

# field size determination
def calcfieldSize(rmap, rmaptrial, thresholdrate=0.5, peakfallTh=0.15, fieldpeakFr=0.5, fieldcutoff=5, pixel2cm=4, L=314):
    rmap = rmap - np.nanmin(rmap)
    placemap = np.zeros(len(rmap))
    placemap[np.where(rmap >= thresholdrate)] = 1 #same as mosers' 1hz threshold
    placemap = measure.label(placemap)
    placemap = measure.label(placemap, background=0)
    numfields = np.max(placemap) #find num of fields
    placeFieldsPeakFr = []
    placeFieldsCenter = []
    placeFieldsSize = []    
    ##lets try to find center of the peaks
    for i in range(1,numfields+1):
        fieldPixels = np.where(placemap==i)
        fieldpeak = max(rmap[fieldPixels])
        #fall off of firing rate to 15% PFR field
        fieldthreshold = max(peakfallTh*fieldpeak, thresholdrate/2.)
        #indixes where firing rate is less than 15%
        fieldsizecutoffind = np.where(rmap[fieldPixels]<fieldthreshold)
        #remove fields samller than 15cm and field peak firing rate less than 
        #1hz and field PFR less than 15% of the max firing rate of the neuron
        if len(fieldPixels[0])<fieldcutoff/pixel2cm or fieldpeak<fieldpeakFr:
            placemap[fieldPixels]=0
        #shorten the field size using the 15% fall off threshold
        if len(fieldsizecutoffind[0])>0:
            placemap[fieldPixels[0][fieldsizecutoffind[0]]]=0
            fieldsizecutoffind1 = np.where(rmap[fieldPixels]>=fieldthreshold)
            fieldPixels = np.array([fieldPixels[0][fieldsizecutoffind1[0]]]) 
    #retiteration after shortening things downs
    placemap = measure.label(placemap)
    placemap = measure.label(placemap, background=0)
    numFields = np.max(placemap)
    newplacemap = np.zeros(placemap.shape)
    #save the field peak firing rate, peak indices, size after all the criteria 
    #this can be improved further
    fieldnum = 0
    for i in np.arange(1,numFields+1):
        fieldPixels = np.where(placemap==i)
        fieldpeak = max(rmap[fieldPixels])
        # check if the cell fire in more than 1/3 of the total laps
        # and is greater than 10 laps
        countlaps=0
        for rt in rmaptrial:
            lappeakfr = np.nanmax(rt[fieldPixels])
            if lappeakfr>=fieldpeak*0.5 and lappeakfr<=fieldpeak*1.5:
                countlaps+=1
        if countlaps>=rmaptrial.shape[0]//3 or countlaps>=15:
            ind = np.where(rmap[fieldPixels]==fieldpeak)
            fieldpeakind = fieldPixels[0][ind[0]][0] 
            placeFieldsPeakFr.append(fieldpeak)
            placeFieldsCenter.append(fieldpeakind)
            placeFieldsSize.append(round(len(fieldPixels[0])*pixel2cm,2)) #converting to cm
            fieldnum = fieldnum + 1
            newplacemap[fieldPixels] = fieldnum
    numFields = len(placeFieldsCenter)
    placeFieldsPeakFr = np.array(placeFieldsPeakFr)
    placeFieldsCenter = np.array(placeFieldsCenter)
    placeFieldsSize = np.array(placeFieldsSize)
    if numFields:
        pfDispersion = getFieldDispersion(rmaptrial, newplacemap, placeFieldsCenter, numFields, cmconversion=pixel2cm, L=L)
    else:
        pfDispersion = np.nan
    return newplacemap, placeFieldsPeakFr, placeFieldsCenter, placeFieldsSize, numFields, pfDispersion

# calculate place field dispersion
def getFieldDispersion(rmaptrial, plmap, pfcenter, nfields, cmconversion=4, L=314):
    M = rmaptrial.shape[0]
    N = rmaptrial.shape[1]
    placeFieldDispersion = []
    for i in range(nfields):
        C = pfcenter[i]*cmconversion
        FieldDis = 0
        dispersionPF = 0
        fieldIndex = np.where(plmap==i+1)[0]
        for m in range(M):
            com = scnd.measurements.center_of_mass(rmaptrial[m,fieldIndex])
            com = com[0]+fieldIndex[0]
            C_i = com*cmconversion
            if not np.isnan(C_i):
                FieldDis += (C - C_i)**2
        dispersionPF = (L/N)*np.sqrt((1./M)*FieldDis)
        placeFieldDispersion.append(dispersionPF)
    return np.array(placeFieldDispersion)

# fuction to get corrected channel map to account for missing channels
def loadCorrectedChanMap(ephysdname, dname, clusterId, df, channelpos, MAX_CHAN=256):
    cmap = np.load(os.path.join(ephysdname, dname, 'channel_map.npy'))[0]
    cpos = np.load(os.path.join(ephysdname, dname, 'channel_positions.npy'))
    if len(cmap)!=MAX_CHAN:
        cmapIdx = []
        for cid in clusterId:
            # find chanel position for the current cluster
            chnum = int(df[df.cluster_id==cid]['ch'])
            cposc = cpos[np.where(cmap==chnum)[0],:]
            # find index for channel num with default channel map
            idx = np.argwhere(np.all(channelpos==cposc, axis=1))[0]
            cmapIdx.append(idx[0])
        cmapIdx = np.array(cmapIdx)
    else:
        cmapIdx = cmap
    return cmapIdx, cmap

# function to calculate highest power theta channel
def calcHighThetaChan(lfp, chanmap, Fs=3000.0):
    pool = mp.Pool(16)
    thetaRatio = pool.map(getBestThetaChannel, [row for row in lfp])
    thetaRatio = np.array(thetaRatio)
    ch = np.argmax(thetaRatio)
    dorsalHCtheta = -mea.get_bandpass_filter_signal(lfp[ch], Fs, [6,10])
    pool.close()
    return dorsalHCtheta, chanmap[ch]

# find best theta channel for a given recording
def getBestThetaChannel(lfp_data, fs=3000.0, f_band=(6,10)):
    P, F, _ = mea.get_spectrogram(lfp_data,fs)
    return np.nanmean(mea.get_relative_power(P, F), f_band=f_band)

# find phase for each spike
def calcSpikePhase(thetapeaktime,spiketimestamps):
    #variable to hold spike phase
    spikephase = [] 
    #iterate over each spike
    for i in range(0,len(spiketimestamps)):
        #assign phase=nan if it does not cross the threshold
        #the second and condition is to account for that the spike occured after the 0th theta peak timestamps or before the last peak timestamps
        if spiketimestamps[i]>=thetapeaktime[0] and spiketimestamps[i]<=thetapeaktime[-1]:
            #find before and after thetapeak timestamps nearest to the ith spike
            _, thetaPeakBefore = find_le(thetapeaktime,spiketimestamps[i])
            _, thetaPeakAfter = find_ge(thetapeaktime,spiketimestamps[i])
            #calculate the interpeak diff
            interpeak_diff = thetaPeakAfter - thetaPeakBefore
            #if theta peak after and theta peak before are same assign spike phase=0
            if thetaPeakAfter==thetaPeakBefore:
                spikephase.append(0.0)
            #this is to account for the fact that the theta peak interval is within 1/6hz to 1/12 i.e. 0.16 to 0.08
            #added a epsilon for now 0.2 to 0.05
            #everyone has to play with this as per their data
            elif 0.08 < interpeak_diff < 0.2:
                #calculate by linear interpolation
                # phase = ((spiketime - thetapeakbefore)/inter_peak_interval)*360
                phase = round((float(spiketimestamps[i]-thetaPeakBefore)/interpeak_diff)*360,3)
                spikephase.append(phase) 
            #if the inter peak difference is outside assign nan
            else:
                spikephase.append(np.nan)
        #phase=nan did not cross threshold
        else:
            spikephase.append(np.nan)
    return np.array(spikephase)

# function to spike phase data 
def calcThetaPeaks(filename, chnum, sessionStart, sessionEnd, f_range=(6,10), fsl=3000.0, ampth=0.7):
    # load lFP data
    dtl=1./fsl
    lfpsig = np.load(filename, mmap_mode='r')
    eeg_sig = lfpsig[chnum,:]
    del lfpsig
    eeg_times = np.array(np.arange(0, len(eeg_sig)*dtl, dtl), dtype='float32')
    behav_end_idx = np.where(eeg_times<=sessionEnd)[0][-1]
    behav_start_idx = np.where(eeg_times>=sessionStart)[0][0]
    eeg_sig = eeg_sig[behav_start_idx:behav_end_idx]
    eeg_times = eeg_times[behav_start_idx:behav_end_idx]
    
    # filter in theta range 6-10hz
    theta_eeg_sig = mea.get_bandpass_filter_signal(eeg_sig,fsl,f_range)
    #amplitude threshold for peak detection. taken directly from Sachin's code
    ampthresh = np.mean(theta_eeg_sig) + ampth*np.std(theta_eeg_sig)
    #detect peak indices. min peak height = ampthresh, min peak distance = fs/10
    theta_peakindices = detect_peaks(theta_eeg_sig, mph=ampthresh, mpd=int(fsl/10.), show=False)
    #calculate the amplitude and time for theta peak
    thetapeaktime = eeg_times[theta_peakindices]
    return thetapeaktime, eeg_times

# calculate TMI
def calcTMI(phase, bins=np.arange(0,4*360+15,15)):
    phase = np.array(phase)
    phase = np.concatenate((phase,phase+360,phase+2*360,phase+3*360,phase+4*360))
    count, edges = np.histogram(phase, bins)
    count = scnd.gaussian_filter1d(count,2)
    count = np.divide(count, np.nanmax(count))
    count2 = count[np.where((edges>=360) & (edges<=720))[0]]
    tmi = 1 - np.nanmin(count2)
    return tmi, count, edges

# function to generate analysis report
def genAnalysisReport(animaldat, HALLWAYS, cid, rewardLocs, xt, xtcm,  colors, linecolors, posMin=0, posMax=314, binwidth=3.14):
    fig, ax = plt.subplots(nrows=6, ncols=4, figsize=(20,10))
    for h,hnum in enumerate(HALLWAYS):
        # trajectory plot
        ax[0][h].plot(animaldat[hnum]['posX'], animaldat[hnum]['posT'], color='k', linewidth=1, rasterized=True)
        ax[0][h].scatter(animaldat['spikedata'][cid][hnum]['spkPos'], animaldat['spikedata'][cid][hnum]['spkPosts'], color=linecolors[hnum], s=10, rasterized=True)
        # occ map
        ax[1][h].imshow(animaldat[hnum]['omaptrsm'], aspect='auto', cmap=colors[hnum], rasterized=True)
        ax[1][h].axvline(x=rewardLocs[hnum][0]*posMax//binwidth, c='k', linestyle='--')
        ax[1][h].axvline(x=rewardLocs[hnum][1]*posMax//binwidth, c='k', linestyle='--')
        # spike map
        ax[2][h].imshow(animaldat['spikedata'][cid][hnum]['spkmaptrsm'], aspect='auto', cmap=colors[hnum], rasterized=True)
        ax[2][h].axvline(x=rewardLocs[hnum][0]*posMax//binwidth, c='k', linestyle='--')
        ax[3][h].axvline(x=rewardLocs[hnum][1]*posMax//binwidth, c='k', linestyle='--')
        # rate map
        ax[3][h].imshow(animaldat['spikedata'][cid][hnum]['rmaptrsm'], aspect='auto', cmap=colors[hnum], rasterized=True)
        ax[3][h].axvline(x=rewardLocs[hnum][0]*posMax//binwidth, c='k', linestyle='--')
        ax[3][h].axvline(x=rewardLocs[hnum][1]*posMax//binwidth, c='k', linestyle='--')
        # mean over occ, spike, and ratemap plot
        ax[1][3].plot(np.nanmean(animaldat[hnum]['omaptrsm'],0), color=linecolors[hnum], rasterized=True)
        ax[2][3].plot(np.nanmean(animaldat['spikedata'][cid][hnum]['spkmaptrsm'],0), color=linecolors[hnum])
        ax[3][3].plot(np.nanmean(animaldat['spikedata'][cid][hnum]['rmaptrsm'],0), color=linecolors[hnum])
        # theta phase precession plot for HC phase
        phase = animaldat['spikedata'][cid][hnum]['spkPhaseHC']
        pos = animaldat['spikedata'][cid][hnum]['spkPos']
        pos = np.concatenate((pos,pos))
        phase = np.concatenate((phase, phase+360))
        ax[4][h].scatter(pos, phase, color=linecolors[hnum], s=1, rasterized=True)
        ax[4][h].axvline(x=rewardLocs[hnum][0]*posMax, c='k', linestyle='--')
        ax[4][h].axvline(x=rewardLocs[hnum][1]*posMax, c='k', linestyle='--')
        # theta phase precession plot for self
        phase = animaldat['spikedata'][cid][hnum]['spkPhaseSelf']
        pos = animaldat['spikedata'][cid][hnum]['spkPos']
        pos = np.concatenate((pos,pos))
        phase = np.concatenate((phase, phase+360))
        ax[5][h].scatter(pos, phase, color=linecolors[hnum], s=1, rasterized=True)
        ax[5][h].axvline(x=rewardLocs[hnum][0]*posMax, c='k', linestyle='--')
        ax[5][h].axvline(x=rewardLocs[hnum][1]*posMax, c='k', linestyle='--')
        # overall tmi
        ax[4][3].plot(animaldat['spikedata'][cid][hnum]['phasebins'][:-1], animaldat['spikedata'][cid][hnum]['tmicounthc'], color=linecolors[hnum], rasterized=True)
        ax[5][3].plot(animaldat['spikedata'][cid][hnum]['phasebins'][:-1], animaldat['spikedata'][cid][hnum]['tmicountself'], color=linecolors[hnum], rasterized=True)
        # label settings
        if hnum==1:
            ax[0][h].set_ylabel('Time (s)', fontsize=18)
            ax[1][h].set_ylabel('Trial #', fontsize=18)
            ax[2][h].set_ylabel('Trial #', fontsize=18)
            ax[3][h].set_ylabel('Trial #', fontsize=18)
            ax[4][h].set_ylabel('Phase (deg)', fontsize=18)
            ax[5][h].set_ylabel('Phase (deg)', fontsize=18)
        # axis limit and tick labels settings
        ax[1][h].set_xticks(xt), ax[1][h].set_xticklabels(xtcm)
        ax[1][3].set_xticks(xt), ax[1][3].set_xticklabels(xtcm)
        ax[2][h].set_xticks(xt), ax[2][h].set_xticklabels(xtcm)
        ax[2][3].set_xticks(xt), ax[2][3].set_xticklabels(xtcm)
        ax[3][h].set_xticks(xt), ax[3][h].set_xticklabels(xtcm)
        ax[3][3].set_xticks(xt), ax[3][3].set_xticklabels(xtcm)
        ax[4][h].set_xticks(xtcm)
        ax[5][h].set_xticks(xtcm)
        ax[0][h].set_xlim([0,314])
        ax[1][h].set_xlim([0,100])
        ax[2][h].set_xlim([0,100])
        ax[3][h].set_xlim([0,100])
        ax[4][h].set_xlim([0,314])
        ax[5][h].set_xlim([0,314])
        ax[0][h].invert_yaxis()
    ax[1][3].set_xlim([0,100])
    ax[2][3].set_xlim([0,100])
    ax[3][3].set_xlim([0,100])
    # firing statistics for each cell and hallway
    sth1 = 'pk:'+str(round(np.max(np.nanmean(animaldat['spikedata'][cid][1]['rmaptrsm'],0)),1))+', m:'+str(round(np.nanmean(np.nanmean(animaldat['spikedata'][cid][1]['rmaptrsm'],0)),1))+', i:'+str(round(animaldat['spikedata'][cid][1]['sinfo'],1))
    sth2 = 'pk:'+str(round(np.max(np.nanmean(animaldat['spikedata'][cid][2]['rmaptrsm'],0)),1))+', m:'+str(round(np.nanmean(np.nanmean(animaldat['spikedata'][cid][2]['rmaptrsm'],0)),1))+', i:'+str(round(animaldat['spikedata'][cid][2]['sinfo'],1))
    sth28 = 'pk:'+str(round(np.max(np.nanmean(animaldat['spikedata'][cid][28]['rmaptrsm'],0)),1))+', m:'+str(round(np.nanmean(np.nanmean(animaldat['spikedata'][cid][28]['rmaptrsm'],0)),1))+', i:'+str(round(animaldat['spikedata'][cid][28]['sinfo'],1))
    ax[0][3].text(0.01,0.9,'Hall#1: ' + sth1)
    ax[0][3].text(0.01,0.6,'Hall#2: ' + sth2)
    ax[0][3].text(0.01,0.3,'Hall#28: ' + sth28)
    ax[0][3].set_axis_off()
    plt.suptitle('Cluster: ' + str(cid), fontsize=20)
    plt.tight_layout()
    return fig
