import os
import rmaputils 
import numpy as np
import pandas as pd
import pylab as plt
import scipy.io as spio
from matplotlib.backends.backend_pdf import PdfPages

# THRESHOLD PARAMS
speedTh = 2 # cm/s
MAX_CHAN = 256

# HALLWAY information
HALLWAYS = [1,2,28] # hallways numbers, Classroom: 1, Sunset:2, Universal: 28
rewardLocs = {1:[0.2,0.65], 2:[0.45,0.83], 28:[0.35,0.83]}
colors = {1:'Blues', 2:'Oranges', 28:'Reds'}
linecolors = {1:'b', 2:'darkorange', 28:'r'}

# POS variables
fspos = 30.
dt = 1./fspos
posMin = 0
posMax = 314 #cm
binwidth = 314/100 #cm
gsmooth = 0.25
bin2cm = binwidth
xt = np.array([0,50,100])
xtcm = xt*binwidth

# LFP variables
fslfp = 1000.0
f_theta = [6,10]
amplfpth = 0.7
phasebins = np.arange(0,4*360+10,10)

# read csv with start and end time for each experimental animal
ephysdirname = r'X:\SWIL-Exp-Rajat\Spikesorted-SWIL'
behavdirname = r'X:\SWIL-Exp-Rajat\Behavior-SWIL'
lfpdirname = r'X:\SWIL-Exp-Rajat\LFP-SWIL'
rmapdirname = r'X:\SWIL-Exp-Rajat\Ratemaps-SWIL'
epochsfname = 'swil-animals.csv'
# load default channel map
rezdat = spio.loadmat(r'X:\SWIL-Exp-Rajat\analysis-scripts\postProcessingKS2\chanMap_intan256F_bottom_SM.mat')
xcoords = np.ravel(rezdat['xcoords'])
ycoords = np.ravel(rezdat['ycoords'])
channelpos = np.array(np.stack((xcoords,ycoords)).T, dtype='int16')
refHCChnum = 20
del rezdat
# load each recording epochs file
epochsdf = pd.read_csv(os.path.join(ephysdirname, epochsfname))
filename = epochsdf['file_name']
start_time, end_time = epochsdf['start_time'], epochsdf['end_time']
# load pooled unit metrics for all animals
pooledMetricsdf = pd.read_csv(os.path.join(ephysdirname,'analyzedMetrics',
                                           'pooledMetricsAllAnimals.csv')).drop(['Unnamed: 0'],axis=1)

# loop through all recordings from each animal
for dname, st, et in zip(filename, start_time, end_time):
    print('.........')
    print('Processing ' + str(dname))
    if 'PPC' in dname:
        aname = dname[:-3]
    else:
        aname = dname[:-2]
    animaldat = {} # save each animals data 
    
    # load behavior data
    behavdata = np.load(os.path.join(behavdirname,aname+'-behavop.npy'), allow_pickle=True).item()
    # iterate through the file to analyze data through each hallway
    for hallwaynum in HALLWAYS:
        behavd = behavdata[hallwaynum]
        
        # load the hallway trial data
        behavd = behavd['trial_data'] 
        posX, posT, posSpeed, trStart, trEnd, omaptr = rmaputils.loadOccupancy(behavd, posMin=posMin, 
                                                                                  posMax=posMax, binwidth=binwidth, 
                                                                                  speedThr=speedTh, dt=dt)
        # create occupancy map
        omaptrsm, omaptrnorm, omap1d, omap1dsm, omapbins = rmaputils.processOccupancy(omaptr, posX, 
                                                                                      posMin=posMin, posMax=posMax, 
                                                                                      binwidth=binwidth, smwindow=gsmooth)
        # save hallway occupancy data for each animal
        animaldat[hallwaynum] = {}
        animaldat[hallwaynum]['posX'] = posX
        animaldat[hallwaynum]['posT'] = posT
        animaldat[hallwaynum]['posSpeed'] = posSpeed
        animaldat[hallwaynum]['trStart'] = trStart
        animaldat[hallwaynum]['trEnd'] = trEnd
        animaldat[hallwaynum]['omaptr'] = omaptr
        animaldat[hallwaynum]['omaptrsm'] = omaptrsm
        animaldat[hallwaynum]['omaptrnorm'] = omaptrnorm
        animaldat[hallwaynum]['omap1d'] = omap1d
        animaldat[hallwaynum]['omap1dsm'] = omap1dsm
        animaldat[hallwaynum]['omapbins'] = omapbins
    del posX, posT, posSpeed, trStart, trEnd, omaptr, omaptrsm, omaptrnorm, omap1d, omap1dsm, omapbins
    
    # load spiketimes and cluster id for good unit for a given recording
    metricsAnimaldf = pooledMetricsdf[pooledMetricsdf.aname==dname]
    spiketimes, clusterId = rmaputils.loadGoodSpikeTimes(os.path.join(ephysdirname,dname), metricsAnimaldf)
    
    # load channel indices for lfp data and lfp filename
    channelIdx, channelmap = rmaputils.loadCorrectedChanMap(ephysdirname, dname, clusterId, metricsAnimaldf, channelpos)
    lfpfilename = os.path.join(lfpdirname, aname+'_lfp.npy')
    # load theta peak times for reference hippocampus electrode
    refThetaPeaks, lfpts = rmaputils.calcThetaPeaks(lfpfilename, refHCChnum, st, et, f_range=f_theta, fsl=fslfp, ampth=amplfpth)
        
    # iterate through spike timestamps
    spikedata = {}
    for cid, spikets, chnum, chidx in zip(clusterId, spiketimes, metricsAnimaldf.ch, channelIdx):
        if 'PPC' in dname:
            chnum = int(chnum + 256) # add to account for PPC channels being at the end
        else:
            chnum = int(chidx)
        # load LFP from reference hc + current unit channel
        spkPhaseRefHC = rmaputils.calcSpikePhase(refThetaPeaks, spikets)
        selfThetaPeaks, _ = rmaputils.calcThetaPeaks(lfpfilename, chnum, st, et, f_range=f_theta, fsl=fslfp, ampth=amplfpth)
        spkPhaseRefSelf = rmaputils.calcSpikePhase(selfThetaPeaks, spikets)
        
        celldata = {} # dict to hold spike data for each neurons in each hallway
        for hallwaynum in HALLWAYS:
            celldata[hallwaynum] = {} 
            # calculate spike pos data
            spkPos, spkPosts, spkPosSpeed, spkPhaseHC, spkPhaseSelf, spkPosTrial, spkPostsTrial = rmaputils.processSpikePos(spikets, spkPhaseRefHC, spkPhaseRefSelf, 
                                                                                                                            animaldat[hallwaynum]['trStart'], 
                                                                                                                            animaldat[hallwaynum]['trEnd'],  
                                                                                                                            animaldat[hallwaynum]['posX'], 
                                                                                                                            animaldat[hallwaynum]['posT'], 
                                                                                                                            animaldat[hallwaynum]['posSpeed'],
                                                                                                                            lfpts)
            # calcualte spike map
            spkmaptr, spkmaptrsm, spkmaptrnorm, spkmap1d, spkmap1dsm = rmaputils.processSpikeMap(spkPosTrial, spkPos, 
                                                                                                 animaldat[hallwaynum]['omapbins'], posMin=posMin, 
                                                                                                 posMax=posMax, binwidth=binwidth, 
                                                                                                 smwindow=gsmooth)
            # calculate ratemaps
            rmaptr, rmaptrsm, rmaptrnorm, rmap1d, rmap1dsm, _ = rmaputils.processRateMap(spkmaptr, animaldat[hallwaynum]['omaptr'], spkmaptrsm, 
                                                                                                animaldat[hallwaynum]['omaptrsm'], spkmap1d, 
                                                                                                animaldat[hallwaynum]['omap1d'], spkmap1dsm, 
                                                                                                animaldat[hallwaynum]['omap1dsm'], fs=fspos)
            # firing statistics: spatial information, sparsity, stability index
            sinfo = rmaputils.calcSpatialInformationScore(rmap1dsm, animaldat[hallwaynum]['omap1dsm'])
            sparsity, _ = rmaputils.calcSparsity(rmap1dsm, rmaptrsm)
            stabilityInd = rmaputils.calcStabilityIndex(rmaptrsm)
            # spatialinfo significance
            sinfo_p, shuffledinfo = rmaputils.calcShuffleSpatialInfo(sinfo, spikets, animaldat[hallwaynum]['omap1dsm'], animaldat[hallwaynum]['posT'][-1], 
                                                                     animaldat[hallwaynum]['posT'][0], animaldat[hallwaynum]['trStart'], animaldat[hallwaynum]['trEnd'], 
                                                                     animaldat[hallwaynum]['posX'], animaldat[hallwaynum]['posT'], animaldat[hallwaynum]['posSpeed'],  
                                                                     animaldat[hallwaynum]['omapbins'], posMin=posMin, posMax=posMax, binwidth=binwidth, fs=fspos)
            # get placefield statistics: num field, field peak firing rate, size, center, dispersion
            if sinfo_p<0.01:
                pfmap, pfPeakFr, pfCenter, pfSize, pfNumFields, pfDispersion = rmaputils.calcfieldSize(rmap1dsm, rmaptrsm, pixel2cm=bin2cm, L=posMax)
            else:
                pfmap, pfPeakFr, pfCenter, pfSize, pfNumFields, pfDispersion = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            # calculate tmi 
            tmihc, tmicounthc, _ = rmaputils.calcTMI(spkPhaseHC, bins=phasebins)
            tmiself, tmicountself, tmiedges = rmaputils.calcTMI(spkPhaseSelf, bins=phasebins)
            
            # store data for each cell in each hallway
            celldata[hallwaynum]['spkPos'] = spkPos
            celldata[hallwaynum]['spkPosts'] = spkPosts
            celldata[hallwaynum]['spkPosSpeed'] = spkPosSpeed
            celldata[hallwaynum]['spkPhaseHC'] = spkPhaseHC
            celldata[hallwaynum]['spkPhaseSelf'] = spkPhaseSelf
            celldata[hallwaynum]['spkPosTrial'] = spkPosTrial
            celldata[hallwaynum]['spkPostsTrial'] = spkPostsTrial
            celldata[hallwaynum]['spkmaptr'] = spkmaptr
            celldata[hallwaynum]['spkmaptrsm'] = spkmaptrsm
            celldata[hallwaynum]['spkmaptrnorm'] = spkmaptrnorm
            celldata[hallwaynum]['spkmap1d'] = spkmap1d
            celldata[hallwaynum]['spkmap1dsm'] = spkmap1dsm
            celldata[hallwaynum]['rmaptr'] = rmaptr
            celldata[hallwaynum]['rmaptrsm'] = rmaptrsm
            celldata[hallwaynum]['rmap1d'] = rmap1d
            celldata[hallwaynum]['rmap1dsm'] = rmap1dsm
            celldata[hallwaynum]['sinfo'] = sinfo
            celldata[hallwaynum]['sparsity'] = sparsity
            celldata[hallwaynum]['stabilityInd'] = stabilityInd
            celldata[hallwaynum]['sinfo_p'] = sinfo_p
            celldata[hallwaynum]['shuffledinfo'] = shuffledinfo
            celldata[hallwaynum]['pfmap'] = pfmap
            celldata[hallwaynum]['pfPeakFr'] = pfPeakFr
            celldata[hallwaynum]['pfCenter'] = pfCenter
            celldata[hallwaynum]['pfSize'] = pfSize
            celldata[hallwaynum]['pfNumFields'] = pfNumFields
            celldata[hallwaynum]['pfDispersion'] = pfDispersion
            celldata[hallwaynum]['tmihc'] = tmihc
            celldata[hallwaynum]['tmiself'] = tmiself
            celldata[hallwaynum]['tmicounthc'] = tmicounthc
            celldata[hallwaynum]['tmicountself'] = tmicountself
            celldata[hallwaynum]['phasebins'] = tmiedges
        # plot the data across hallways for this cell
        spikedata[cid] = celldata
        del celldata
    animaldat['spikedata'] = spikedata
    del spikedata
    # save the processed data
    opfname = os.path.join(rmapdirname, dname+'-op.npy')
    np.save(opfname, animaldat)
    # plot the output pdf
    pdfname = os.path.join(rmapdirname, dname+'-op.pdf')
    with PdfPages(pdfname) as pdf: 
        for cid in clusterId:
            fig = rmaputils.genAnalysisReport(animaldat, HALLWAYS, cid, rewardLocs, xt, xtcm, colors, linecolors, posMin=posMin, posMax=posMax, binwidth=binwidth)
            pdf.savefig(fig, dpi=100)
            plt.close()
