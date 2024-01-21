import os
import math
import glob
import cupy as cp
import numpy as np
import scipy.signal as spsig
from natsort import natsorted
from intanutil.read_data import read_data


def downsample(factors, sig):
    '''
    Avoids NaNs by calling decimate multiple times:
    docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html
    '''
    for f in factors:
        fx = spsig.decimate(arr, f)
        sig = np.apply_along_axis(fx, 1, sig)
    return sig

def channel_shift(data, sample_shifts):
    """
    GPU shifts signals via a Fourier transform to correct for diff sampling times
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
    subsample_factors = [5, 6]
    subsample_total = numpy.prod(subsample_factors)
    use_default = input(f"Use the default downsampling factor of {subsample_total}? (y/n) ")
    use_default= ('y' in use_default) or ('Y' in use_default)
    if not default_factor:
        subsample_total = int(input("What downsampling factor would you like to use then? "))
        # see downsample() for reasoning
        if subsample_total > 13:
            power = math.ceil(subsample_total / 13)
            help_txt = f"Downsampling by this much requires smaller steps. " 
            help_txt += f"List {power} factors separated by spaces that multiply to {subsample_total}. "
            help_txt += 'E.g. "5, 6" for a downsampling factor of 30'
            subsample_factors = input(help_txt).split().replace("'", "").replace('"', '')
            subsample_factors = [int(x) for x in subsample_factors]
            tot = np.prod(subsample_factors)
            err_txt = f'{subsample_factors} multiply to {tot}, not {subsample_total}'
            assert tot == subsample_total, err_txt
        
    ### @TODO try to autodetect .rhd files in the current directory. If found, list them
    ### @TODO if not found then input(rhd_directory) and input(save_dir), make_dirs(save_dir, exist_ok=True)
    dirname = '/media/rajat/mcnlab_store2/Research/SPrecordings/Rajat_Data/Data-Enrichment/EERound2/ET2'
    rawfname = 'ET2_211228_174841'
    save_dir = input('Where would you like to save the output?')
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.abspath(save_dir)
    
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
    dig_in = np.array([])
    for i, filename in enumerate(files):
        filename = os.path.basename(filename)
        print("\n ***** Loading: " + filename)
        rhd_path = os.path.join(dirname,rawfname,filename)
        ts, amp_data, digIN, analogIN, fs = read_data(rhd_path) 
        if saveAnalog:
            analog_in = np.concatenate((analog_in, analogIN[0]), dtype=np.float32)
        else:
            del analogIN
        amp_data_n  = []
        for c in range(num_ch):
            shifted = channel_shift([amp_data[c]], [shift[c]])
            shifted_offset = np.array(shifted[0] - 32768, dtype=np.int16)
            amp_data_n.append(shifted_offset)
        del amp_data
        amp_data_n = np.array(amp_data_n)
        shifted_path = os.path.join(save_dir,filename[:-4]+'_shifted.bin')
        arr = np.memmap(shifted_path, dtype='int16', mode='w+', shape=amp_data_n.T.shape)
        arr[:] = amp_data_n.T
        del arr
        if saveLFP:
            # convert microvolts for lfp conversion
            amp_data_n = np.multiply(0.195,  amp_data_n, dtype=np.float32)
            print("REAL FS = " + str(1 ./ np.nanmedian(np.diff(ts))))
            starts = ts[-1]+1 ./ fs
            size = amp_data_n.shape[1]
            if i == 0:
                startind = 0
                ind = np.arange(0, size, subsample_factor)
            else:
                startind = np.where(ts>=starts)[0][0]
                ind = np.arange(startind, size, subsample_factor)
            amp_data_n = downsample(subsample_factors, amp_data_n[:,startind:])
            amp_data_mmap = np.concatenate((amp_data_mmap, amp_data_n), 1)
            dig_in = np.concatenate((dig_in, digIN))).astype(np.uint8)
            amp_ts_mmap = np.concatenate((amp_ts_mmap, ts))
        del amp_data_n

    if saveAnalog:
        np.save(analogIn_filename, analog_in)
    if saveLFP:
        np.save(lfp_filename, amp_data_mmap)
        np.save(lfpts_filename, amp_ts_mmap)
        np.save(digIn_filename, dig_in)
