import os
import math
import glob
import cupy as cp
import numpy as np
import scipy.signal as spsig
from natsort import natsorted
from intanutil.read_data import read_data
from intanutil.read_header import read_header


def downsample(factors, sig):
    '''
    Avoids NaNs by calling decimate multiple times:
    docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html
    '''
    for f in factors:
        fx = lambda sig : spsig.decimate(sig, f)
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
    subsample_total = np.prod(subsample_factors)
    print()
    use_default = input(f"Use the default downsampling factor of {subsample_total}? (y/n) ")
    use_default= ('y' in use_default) or ('Y' in use_default)
    if not use_default:
        print()
        subsample_total = int(input("What downsampling factor would you like to use then? "))
        # see downsample() for reasoning
        if subsample_total > 13:
            power = math.ceil(np.emath.logn(13, subsample_total))
            print()
            help_txt = 'Downsampling by this much requires smaller steps. '
            help_txt += f'List {power} factors separated by commas that multiply to {subsample_total}. '
            help_txt += 'E.g. 5, 6 for a downsampling factor of 30 \n\n'
            subsample_factors = input(help_txt).split(',')
            subsample_factors = [int(x) for x in subsample_factors]
            tot = np.prod(subsample_factors)
            err_txt = f'{subsample_factors} multiply to {tot}, not the specified {subsample_total}'
            assert tot == subsample_total, err_txt

    print()
    dirs_txt = "You can specify multiple directories to process. " 
    dirs_txt += "It is assumed that each directory has .rhd files for one animal recording. "
    dirs_txt += "It is also assumed that all recordings have the same probe setup. "
    dirs_txt += "List your directories separated by commas. \n"
    dirs_txt += "E.g. C:\\animal1_day1, C:\\animal2_day6 \n\n"
    dirs = input(dirs_txt).replace(", ", ",").split(',')
    # in case someone is silly enough to append a trailing comma
    if dirs[-1] == "":
        del dirs[-1]
    for d in dirs:
        assert os.path.exists(d), f'Recording directory {d} could not be found :('
    
    save_dir = input('\nWhere would you like to save the outputs? \n\n')
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.abspath(save_dir)

    saveLFP = input('\nWould you like to save the LFP? (y/n) ')
    saveLFP = ('y' in saveLFP) or ('Y' in saveLFP)
    saveAnalog = input('\nWould you like to save the analog signal? (y/n) ')
    saveAnalog = ('y' in saveAnalog) or ('Y' in saveAnalog)

    animals = []
    for d in dirs:
        name = input(f"\nWhat is the animal's ID for {d}?\n\n")
        assert name != "", "names cannot be empty"
        animals.append(name)
   
    files = natsorted(glob.glob(os.path.join(dirs[0], '*.rhd')))
    first_dirs_first_rhd = os.path.join(d, files[0])
    fid = open(first_dirs_first_rhd, 'rb')
    header = read_header(fid)
    num_ch = header['num_amplifier_channels']
    shift = np.tile(np.linspace(-1,0,32),num_ch // 32)
    print()
    multi_roi = f"{num_ch} recording channels found. "
    multi_roi += "You can split these into multiple ROIs (e.g. VC & PCC). "
    multi_roi += "Each ROI gets its own binary file(s). "
    multi_roi += "Would you like to split ? (y/n) "
    multi_roi = input(multi_roi)
    multi_roi = ('y' in multi_roi) or ('Y' in multi_roi)
    # [(naming_prefix, start channel, end channel)]
    roi_s = [("", 0, num_ch-1)]
    if multi_roi:
        roi_s = [] # overwrite / empty
        print()
        num_roi = int(input("How many ROIs were recorded from? "))
        for _i in range(num_roi):
            print()
            roi_name = input(f"What's ROI #{_i+1}'s name? (e.g. VC) ")
            if _i == 0: # only show this message one time
                print("\nChannels are 1-indexed in this script, like in Matlab")
                print("E.g. a 128 channel recording starts on 1 and ends on 128")
            start_ch = int(input(f"\nWhich channel does {roi_name} start? ")) - 1
            end_ch = int(input(f"\nWhich channel does {roi_name} end? ")) - 1
            roi_info = (roi_name, start_ch, end_ch)
            roi_s.append(roi_info)
    # small sanity check that at least first and last channel are included
    first_ch, last_ch = False, False
    for _, st, end in roi_s:
        if st == 0:
            first_ch = True
        if end == num_ch - 1:
            last_ch = True
    err_txt = "channels 0 and {num_ch - 1} not specified. Incorrect user input."
    assert first_ch and last_ch, err_txt

    # ask for user inputs before this long loop if possible!
    overwrite = None
    for animal_id, d in zip(animals, dirs):
        d = os.path.normpath(d)
        sub_save_dir = os.path.join(save_dir, os.path.basename(d))
        os.makedirs(sub_save_dir, exist_ok=True)
        sub_save_dir = os.path.abspath(sub_save_dir)
        lfp_filename = os.path.join(sub_save_dir, animal_id+'-lfp.npy')
        lfpts_filename = os.path.join(sub_save_dir, animal_id+'-lfpts.npy')
        digIn_filename = os.path.join(sub_save_dir, animal_id+'-digIn.npy')
        analogIn_filename = os.path.join(sub_save_dir, animal_id+'-analogIn.npy')
        # can protect data or allow quickly resuming after an error
        old_data = glob.glob(os.path.join(sub_save_dir,'*shifted.bin'))
        if old_data:
            if overwrite == None:
                print()
                overwrite = input('Old binaries found! Overwrite? (y/n) ')
                overwrite = ('y' in overwrite) or ('Y' in overwrite)
            if overwrite == False:
                continue               
    
        starts = 0
        dig_in = np.array([])
        analog_in = np.array([])
        amp_ts_mmap = np.array([])
        amp_data_mmap = np.array([[]] * num_ch)    
        roi_offsets = [0] * len(roi_s) 
        files = natsorted(glob.glob(os.path.join(d, '*.rhd')))
        for i, filename in enumerate(files):
            filename = os.path.basename(filename)
            print("\n ***** Loading: " + filename)
            rhd_path = os.path.join(d, filename)
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
            for r_i, roi in enumerate(roi_s):
                name, start, end = roi
                offset = roi_offsets[r_i]
                roi_data = amp_data_n[start:end+1]
                shifted_path = os.path.join(sub_save_dir, name + '_shifted_merged.bin')
                shape = (roi_data.shape[1] + int(offset / roi_data.shape[0] / 2), roi_data.shape[0])
                m = 'w+'
                if i > 0:
                    m = 'r+' # extend if already created
                arr = np.memmap(shifted_path, dtype='int16', mode=m, shape=shape)
                # update this ROI's binary file offset
                roi_offsets[r_i] += 2 * np.prod(roi_data.shape) 
                # append to the end of the large binary file
                arr[-roi_data.shape[-1]:,:] = roi_data.T
                del arr
            if saveLFP:
                # convert microvolts for lfp conversion
                amp_data_n = np.multiply(0.195, amp_data_n, dtype=np.float32)
                print("REAL FS = " + str(1.0 / np.nanmedian(np.diff(ts))))
                size = amp_data_n.shape[1]
                fs = fs / float(subsample_total)
                if i == 0:
                    start_i = 0
                else:
                    start_i = np.where(ts >= starts)[0][0]
                starts = ts[-1] + 1.0 / fs    
                ind = np.arange(start_i, size, subsample_total)
                ts = ts[ind]
                amp_data_n = downsample(subsample_factors, amp_data_n[:, start_i:])
                amp_data_mmap = np.concatenate((amp_data_mmap, amp_data_n), 1)
                dig_in = np.concatenate((dig_in, digIN)).astype(np.uint8)
                amp_ts_mmap = np.concatenate((amp_ts_mmap, ts))
            del amp_data_n
    
        if saveAnalog:
            np.save(analogIn_filename, analog_in)
        if saveLFP:
            np.save(lfp_filename, amp_data_mmap)
            np.save(lfpts_filename, amp_ts_mmap)
            np.save(digIn_filename, dig_in)
