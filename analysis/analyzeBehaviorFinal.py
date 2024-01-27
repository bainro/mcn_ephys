# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:49:20 2023

@author: jshobe
"""

import os, glob
import numpy as np
import seaborn as sns
from behavutils import *
import scipy.stats as spst
import matplotlib.pyplot as plt

# init variables
c = {'1':[0.2,0.65], '2':[0.45,0.83], '28':[0.35, 0.83]}
Halltitle = {'1': 'Classroom', '2':'Sunset', '28':'Universal'}   
BIN2CM = 3.14 # cm
BINWIDTH = 0.01 # 3cm bins
fs = 30 # behavior sampling rate
radiusVR = 50 # default ball radius
ENVLENGTH = 314 # total env length in cm

# directories to work on
# NOTE the main behavior file had RECORDING.mat in the end
animalsdirname = [r'Y:\Research\SPrecordings\Rajat_Data\Data-SWIL\SWILRound3\SWIL105',
                  r'Y:\Research\SPrecordings\Rajat_Data\Data-SWIL\SWILRound3\SWIL11',
                  r'Y:\Research\SPrecordings\Rajat_Data\Data-SWIL\SWILRound3\SWIL12',
                  r'Y:\Research\SPrecordings\Rajat_Data\Data-SWIL\SWILRound3\SWIL13',
                  r'Y:\Research\SPrecordings\Rajat_Data\Data-SWIL\SWILRound4\SWIL15',
                  r'Y:\Research\SPrecordings\Rajat_Data\Data-SWIL\SWILRound4\SWIL18',
                  r'Y:\Research\SPrecordings\Rajat_Data\Data-SWIL\SWILRound4\SWIL19',
                  r'Y:\Research\SPrecordings\Rajat_Data\Data-SWIL\SWILRound4\SWIL20',
                  r'Y:\Research\SPrecordings\Rajat_Data\Data-SWIL\SWILRound5\SWIL22',
                  r'Y:\Research\SPrecordings\Rajat_Data\Data-SWIL\SWILRound5\SWIL23',
                  r'Y:\Research\SPrecordings\Rajat_Data\Data-SWIL\SWILRound5\SWIL24',
                  r'Y:\Research\SPrecordings\Rajat_Data\Data-SWIL\SWILRound5\SWIL25',
                  r'Y:\Research\SPrecordings\Rajat_Data\Data-SWIL\SWILRound5\SWIL26']
animalsname = ['SWIL105','SWIL11','SWIL12','SWIL13','SWIL15','SWIL18','SWIL19','SWIL20',
               'SWIL22','SWIL23','SWIL24','SWIL25','SWIL26']
# intan start and end alignment, sadly this has to be done manually due to file crashes, etc.
# SHIIIIITTTTTT
indices = [[1,-1], [None,None], [1,92716], [1,87204], [7399,70203], [10942,60279],
           [19264,89879], [942,None], [1,90899], [14406,73197], [9719,77424], [8450,71316],
           [14468,108568]]

# mean speed across all animals
meanSpeedAllAnimals = []
# iterate through all behavior files
for aname,adname,itidx in zip(animalsname,animalsdirname, indices):
    print('Processing '+str(aname)+'...')
    # load behavior data file
    fname = glob.glob(os.path.join(adname,'Behavior','dis_vel_rew_*RECORDING.mat'))
    dat = readBehavFile(fname[0])
    # load intan timestamps
    intanfilename = os.path.join(adname,aname+'_digIn.npy')
    intants = loadintants(intanfilename, stidx=itidx[0], etidx=itidx[-1])
    # load processed data
    if 'SWIL11' in aname:
        dat = processBehavFile(dat, samplerate=fs, VRradius=radiusVR, intants=None)
        dat = dat.iloc[:len(intants)]
        dat['time'] = intants
        vel = np.diff(dat['poscm'])/np.diff(intants)
        vel = np.insert(vel,0,np.nan)
        dat['vel'] = vel
    else:
        dat = processBehavFile(dat, samplerate=fs, VRradius=radiusVR, intants=intants)
    # get unique hallway number
    hallnum = np.unique(np.array(dat['hallnum'], dtype=int))
    hallnum = hallnum[~np.isnan(hallnum)]
    # variable to store final op
    allhallwaydata = {}
    meanSpeed = []
    # iterate over each halls
    for hnum in hallnum:
        # sanity check to skip certain hallways 
        if sum(np.array(dat['hallnum'],dtype=int)==hnum)>5:
            # create hallway specific dataframe
            allhallwaydata[hnum] = calcHallwayData(dat, hnum, binwidth=BINWIDTH)
            normspeed = norm1D(np.nanmean(allhallwaydata[hnum]['binned_speed'],0)[1:])
            print(allhallwaydata[hnum]['binned_speed'].shape[0])
            meanSpeed.append(normspeed)
    meanSpeedAllAnimals.append(meanSpeed)
    # plot all hallways data
#    figname = os.path.join(adname,'Behavior',aname+'-behav.png')
#    plotSpeedAcrossHallways(allhallwaydata, figname, aname, bin2cm=BIN2CM)
    # save the data
    opfname = os.path.join(adname,'Behavior', aname+'-behavop.npy')
    np.save(opfname, allhallwaydata)
    del allhallwaydata
meanSpeedAllAnimals = np.array(meanSpeedAllAnimals)


# plot all animals speed across all environments
xpos = np.linspace(0,314,meanSpeedAllAnimals.shape[-1])
yh1 = np.nanmean(meanSpeedAllAnimals[:,0,:],0)
yh1err = spst.sem(meanSpeedAllAnimals[:,0,:],0, nan_policy='omit')
yh2 = np.nanmean(meanSpeedAllAnimals[:,1,:],0)
yh2err = spst.sem(meanSpeedAllAnimals[:,1,:],0, nan_policy='omit')
yh3 = np.nanmean(meanSpeedAllAnimals[:,2,:],0)
yh3err = spst.sem(meanSpeedAllAnimals[:,2,:],0, nan_policy='omit')
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14,3))
ax[0].plot(xpos, yh1, 'b')
ax[0].fill_between(xpos, yh1-yh1err, yh1+yh1err, color='b', alpha=0.5)
ax[0].axvline(x=0.2*ENVLENGTH, ymin=0, ymax=1, c='k', linestyle='--', linewidth=2)
ax[0].axvline(x=0.65*ENVLENGTH, ymin=0, ymax=1, c='k', linestyle='--', linewidth=2)
ax[0].set_xlim([0,314])
ax[0].set_ylim([0,1])
ax[0].set_yticks([])
ax[0].set_xlabel('Position (cm.)', fontsize=22)
ax[0].set_ylabel('Norm. Speed', fontsize=22)
ax[1].plot(xpos, yh2, 'darkorange')
ax[1].fill_between(xpos, yh2-yh2err, yh2+yh2err, color='darkorange', alpha=0.5)
ax[1].axvline(x=0.45*ENVLENGTH, ymin=0, ymax=1, c='k', linestyle='--', linewidth=2)
ax[1].axvline(x=0.83*ENVLENGTH, ymin=0, ymax=1, c='k', linestyle='--', linewidth=2)
ax[1].set_xlim([0,314])
ax[1].set_ylim([0,1])
ax[1].set_yticks([])
ax[1].set_xlabel('Position (cm.)', fontsize=22)
ax[2].plot(xpos, yh3, 'r')
ax[2].fill_between(xpos, yh3-yh3err, yh3+yh3err, color='r', alpha=0.5)
ax[2].axvline(x=0.35*ENVLENGTH, ymin=0, ymax=1, c='k', linestyle='--', linewidth=2)
ax[2].axvline(x=0.83*ENVLENGTH, ymin=0, ymax=1, c='k', linestyle='--', linewidth=2)
ax[2].set_xlim([0,314])
ax[2].set_ylim([0,1])
ax[2].set_yticks([])
ax[2].set_xlabel('Position (cm.)', fontsize=22)
for ax_ in ax:
    ax_.tick_params(axis='x', labelsize=18)
sns.despine()
plt.tight_layout()
plt.savefig('speedAcrossAnimals.pdf', dpi=200)