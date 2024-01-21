import os
import glob
import cupy as cp
import numpy as np
import scipy.signal as spsig
from natsort import natsorted
from intanutil.read_data import read_data

def decimateSig(arr):
    """function to decimate signal"""
    return spsig.decimate(arr, 5)

def decimateSig2(arr):
    """function to decimate signal"""
    return spsig.decimate(arr, 6)

def channel_shift(data, sample_shifts):
    """
    GPU Shifts channel signals via a Fourier transform to correct for different sampling times
    :param data: cupy array with shape (n_channels, n_times)
    :param sample_shifts: channel shifts, cupy array with shape (n_channels)
    :return: Aligned data, cupy array with shape (n_channels, n_times)
    """
    data = cp.array(data)
    sample_shifts = cp.array(sample_shifts)
    
    n_channels, n_times = data.shape

    dephas = cp.tile(- 2 * np.pi / n_times * cp.arange(n_times), (n_channels, 1))
    dephas += 2 * np.pi * (dephas < - np.pi)  # angles in the range (-pi,pi)
    dephas = cp.exp(1j * dephas * sample_shifts[:, cp.newaxis])

    data_shifted = cp.real(cp.fft.ifft(cp.fft.fft(data) * dephas))

    return cp.asnumpy(data_shifted)

if __name__ == "__main__":
    # this need to be changed for each animal
    subsamplingfactor = 30
    ### @TODO try to autodetect .rhd files in the current directory. If found, list them
    ### @TODO if not found then input(rhd_directory) and input(save_dir), make_dirs(save_dir, exist_ok=True)
    dirname = '/media/rajat/mcnlab_store2/Research/SPrecordings/Rajat_Data/Data-Enrichment/EERound2/ET2'
    rawfname = 'ET2_211228_174841'
    save_dir = input('Where would you like to save the output?')
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.abspath(save_dir)
    ### @TODO LFP & Analog: what's the diff, just downsampling?
    saveLFP = input('Would you like to save the LFP? (y/n) ')
    saveLFP = ('y' in saveLFP) or ('Y' in saveLFP)
    saveAnalog = input('Would you like to save the analog signal? (y/n) ')
    saveAnalog = ('y' in saveAnalog) or ('Y' in saveAnalog)

    aname = input("What's the animal's ID / name?")
    lfp_filename = os.path.join(save_dir, aname+'-lfp.npy')
    lfpts_filename = os.path.join(save_dir, aname+'-lfpts.npy')
    digIn_filename = os.path.join(save_dir, aname+'-digIn.npy')
    analogIn_filename = os.path.join(save_dir, aname+'-analogIn.npy')

    files = natsorted(glob.glob(os.path.join(dirname,rawfname,'*.rhd')))
    amp_data = read_data(os.path.join(dirname,rawfname,files[0]))[1]
    num_ch = amp_data.shape[0]
    # variable to calculate shift
    shift = np.tile(np.linspace(-1,0,32),num_ch // 32)

    analog_in = np.array([])
    amp_ts_mmap = np.array([])
    amp_data_mmap = np.array([])
    for i, filename in enumerate(files):
        filename = os.path.basename(filename)
        print("\n ***** Loading: " + filename)
        ts, amp_data, digIN, analogIN, fs = read_data(os.path.join(dirname,rawfname,filename)) 
        if saveAnalog:
            analog_in = np.concatenate((analog_in, analogIN[0]), dtype=np.float32)

        amp_data_n  = []
        for c in range(amp_data.shape[0]):
            amp_data_n.append(np.array(channel_shift(np.array([amp_data[c]]), np.array([shift[c]]))[0] - 32768, dtype=np.int16))
        del amp_data
        amp_data_n = np.array(amp_data_n)
        arr = np.memmap(os.path.join(save_dir,filename[:-4]+'_shifted.bin'), dtype='int16', mode='w+', shape=amp_data_n.T.shape)
        arr[:] = amp_data_n.T
        del arr
        if saveLFP:
            # convert microvolts for lfp conversion
            amp_data_n = np.multiply(0.195,  amp_data_n, dtype=np.float32)
            print("REAL FS = " + str(1./np.nanmedian(np.diff(ts))))

            size = amp_data_n.shape[1]

            if i == 0:
                ind = np.arange(0,size,subsamplingfactor)
            else:
                startind = np.where(ts>=starts)[0][0]
                ind = np.arange(startind,size,subsamplingfactor)

            ### @TODO automatically split decimate into calls with less than n=13, allowing user input?
            ### @TODO this first line is special for i > 1 ... 
            amp_data_n = np.apply_along_axis(decimateSig,1,amp_data_n[:,startind:])
            amp_data_n = np.apply_along_axis(decimateSig2,1,amp_data_n)
            starts = ts[-1]+1./fs
            amp_data_mmap = np.concatenate((amp_data_mmap, amp_data_n), 1)
            amp_ts_mmap = np.concatenate((amp_ts_mmap, ts))
            del amp_data_n
            if i != 0:
                dig_in = np.array(np.concatenate((dig_in, digIN)), dtype='uint8')
                
                
    if saveLFP:
        np.save(lfp_filename, amp_data_mmap)
        np.save(lfpts_filename, amp_ts_mmap)
        np.save(digIn_filename, dig_in)
        if saveAnalog:
            np.save(analogIn_filename, analog_in)
