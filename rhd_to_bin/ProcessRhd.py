import os
import math
import glob
import time
import threading
import cupy as cp
import numpy as np
import scipy.signal as spsig
from natsort import natsorted
from numpy.lib.format import open_memmap
# fixes "No module named intanutil" err
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
from intanutil.read_data import read_data
from intanutil.read_header import read_header
from intanutil.get_bytes_per_data_block import get_bytes_per_data_block
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
    
    shifted_and_offset = cp.array(data_shifted[0] - 32768, dtype=cp.int16)

    return cp.asnumpy(shifted_and_offset)

def dir_worker(d, roi_s, num_ch, saveLFP, saveAnalog, 
               save_dir, animal_id, subsample_factors):
    subsample_total = np.prod(subsample_factors)
    
    d = os.path.normpath(d)
    if d == save_dir or save_dir == None:
        sub_save_dir = d
    else:
        sub_save_dir = os.path.join(save_dir, os.path.basename(d))
        os.makedirs(sub_save_dir, exist_ok=True)
        sub_save_dir = os.path.abspath(sub_save_dir)
    
    lfp_bin_name = os.path.join(sub_save_dir, animal_id+'-lfp.bin')
    lfp_filename = os.path.join(sub_save_dir, animal_id+'-lfp.npy')
    lfpts_filename = os.path.join(sub_save_dir, animal_id+'-lfpts.npy')
    digIn_filename = os.path.join(sub_save_dir, animal_id+'-digIn.npy')
    digIn_ts_filename = os.path.join(sub_save_dir, animal_id+'-digInts.npy')
    analogIn_filename = os.path.join(sub_save_dir, animal_id+'-analogIn.npy')             

    starts = 0
    dig_in = None
    dig_in_ts = np.array([])
    analog_in = np.array([])
    amp_ts_mmap = np.array([])
    roi_offsets = [0] * len(roi_s)
    lfp_offset = 0
    files = natsorted(glob.glob(os.path.join(d, '*.rhd')))
    if len(files) == 0:
        return
    # User requested file. Removed upon successful completion.
    crash_file = os.path.join(sub_save_dir, 'CRASHED_removed_at_end')
    with open(crash_file, 'w') as _:
        pass
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
            amp_data_n.append(shifted)
        del amp_data
        amp_data_n = np.array(amp_data_n)
        for r_i, roi in enumerate(roi_s):
            name, start, end = roi
            offset = roi_offsets[r_i]
            roi_data = amp_data_n[start:end+1]
            shifted_path = os.path.join(sub_save_dir, name + '_shifted_merged.bin')
            rows, cols = roi_data.shape
            shape = (cols + int(offset / rows / 2), rows) # 16bits == 2bytes
            m = 'w+'
            if i > 0:
                m = 'r+' # extend if already created
            arr = np.memmap(shifted_path, dtype='int16', mode=m, shape=shape)
            # update this ROI's binary file offset
            roi_offsets[r_i] += 2 * np.prod(roi_data.shape, dtype=np.float64) 
            # append to the end of the large binary file
            arr[-cols:,:] = roi_data.T
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
            ind = np.arange(start_i, size, subsample_total)    
            amp_ts = ts[ind]
            starts = amp_ts[-1] + 1.0 / fs
            amp_data_n = downsample(subsample_factors, amp_data_n[:, start_i:])
            if dig_in == None:
                dig_in = digIN
            else:
                dig_in = np.concatenate((dig_in, digIN), 1).astype(np.uint8)
            dig_in_ts = np.concatenate((dig_in_ts, ts))
            amp_ts_mmap = np.concatenate((amp_ts_mmap, amp_ts))
            rows, cols = amp_data_n.shape
            shape = (cols + round(lfp_offset / rows / 4), rows)
            arr = np.memmap(lfp_bin_name, dtype='float32', mode=m, shape=shape)
            lfp_offset += 4 * np.prod(amp_data_n.shape, dtype=np.float64) 
            # append to the end of the large binary file
            arr[-cols:,:] = amp_data_n.T
            del arr
        del amp_data_n

    if saveAnalog:
        np.save(analogIn_filename, analog_in)
    if saveLFP:
        np.save(lfpts_filename, amp_ts_mmap)
        np.save(digIn_ts_filename, dig_in_ts)
        np.save(digIn_filename, dig_in)
        for c in range(num_ch):
            lfp = np.memmap(lfp_bin_name, dtype='float32', mode=m, shape=shape)
            # create a memory-mapped .npy file with the same dimensions and dtype
            m2 = 'w+'
            if c == 0:
                m2 = 'r+'
            npy = open_memmap(lfp_filename, mode=m2, dtype=lfp.dtype, shape=lfp.shape[::-1])
            # copy the array contents
            npy[c,:] = lfp.T[c,:]
            del lfp
            del npy
        os.remove(lfp_bin_name)
        
    # remove CRASHED file to signify processing completion
    os.remove(crash_file)
    log_file = os.path.join(sub_save_dir, 'log.txt')
    # let user know how many RHD files were processed
    with open(log_file, 'w') as log_f:
        log_f.write(f"{len(files)} RHD files processed.")

if __name__ == "__main__":    
    dirs = []
    if gui:
        print("\nA GUI / dialog box should appear in the background. Press 'cancel' or ESC to end.")
        t = "Choose directory(s) with RHD files."
        while True:
            d = filedialog.askdirectory(mustexist=True, title=t)
            # windows ('nt') vs linux
            if os.name == 'nt':
                gui_root.attributes('-topmost', True, '-alpha', 0)
            if d == () or d == '':
                break
            else:
                dirs.append(d)
    else:
        print()
        dirs_txt = "Which directories do you want to process? You may specify multiple. " 
        dirs_txt += "It is assumed that each directory has .rhd files for one animal recording. "
        dirs_txt += "It is also assumed that all recordings have the same probe setup. "
        dirs_txt += "List your directories separated by commas. \n"
        dirs_txt += "E.g. C:\\animal1_day1, C:\\animal2_day6 \n\n"
        dirs = input(dirs_txt).replace(", ", ",").split(',')
        # in case someone is silly enough to append a trailing comma
        if dirs[-1] == "":
            del dirs[-1]
    dirs = list(set(dirs)) # remove duplicates
    assert len(dirs) > 0, "No input directories specified :("
    for d in dirs:
        assert os.path.exists(d), f'Recording directory {d} could not be found :('
    
    if gui:
        print("\nA GUI / dialog box should appear. It might be in the background")
        save_dir = input("\nSave outputs to the same input directory(s)? (y/n) ")
        save_dir = ('y' in save_dir) or ('Y' in save_dir)
        if save_dir:
            save_dir = None
        else:
            save_dir = filedialog.askdirectory(title="Select directory to save outputs")
            # windows ('nt') vs linux
            if os.name == 'nt':
                gui_root.attributes('-topmost', True, '-alpha', 0)
            os.makedirs(save_dir, exist_ok=True)
            save_dir = os.path.abspath(save_dir)
    else:
        save_dir = input("Save outputs to the same input directory(s)? (y/n) ")
        save_dir = ('y' in save_dir) or ('Y' in save_dir)
        if save_dir:
            save_dir = None
        else:
            save_dir = input('\nWhere would you like to save the outputs? \n\n')
            os.makedirs(save_dir, exist_ok=True)
            save_dir = os.path.abspath(save_dir)

    saveLFP = input('\nWould you like to save the LFP? (y/n) ')
    saveLFP = ('y' in saveLFP) or ('Y' in saveLFP)
    saveAnalog = input('\nWould you like to save the analog signal? (y/n) ')
    saveAnalog = ('y' in saveAnalog) or ('Y' in saveAnalog)
    
    subsample_factors = [0]
    if saveLFP:
        subsample_factors = [5, 6]
        subsample_total = np.prod(subsample_factors)
        print()
        use_default = input(f"Use the default LFP downsampling factor of {subsample_total}? (y/n) ")
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
   
    files = natsorted(glob.glob(os.path.join(dirs[0], '*.rhd')))
    first_dirs_first_rhd = os.path.join(d, files[0])
    fid = open(first_dirs_first_rhd, 'rb')
    filesize = os.path.getsize(first_dirs_first_rhd)
    header = read_header(fid)
    # Determine how many samples the data file contains.
    bytes_per_block = get_bytes_per_data_block(header)
    bytes_remaining = filesize - fid.tell()
    num_data_blocks = int(bytes_remaining / bytes_per_block)
    num_amplifier_samples = header['num_samples_per_data_block'] * num_data_blocks
    record_time = round(num_amplifier_samples / header['sample_rate'])
    num_ch = header['num_amplifier_channels']
    shift = np.tile(np.linspace(-1,0,32), num_ch // 32)
    print()
    num_roi = f"{num_ch} recording channels found.\n"
    num_roi += "How many ROIs were recorded from? "
    num_roi = int(input(num_roi))
    # [(naming_prefix, start channel, end channel)]
    roi_s = [("", 0, num_ch-1)]
    if num_roi > 1:
        roi_s = [] # overwrite / empty
        start_ch = 0
        for roi_id in range(num_roi):
            print()
            roi_name = input(f"What's ROI #{roi_id+1}'s name? (e.g. VC) ")
            # for people who don't read the prompt... *cough* winny *cough*
            try:
                roi_num_ch = input(f"\nHow many channels does {roi_name} have? ")
                int(roi_num_ch)
            except:
                roi_num_ch = input(f"\nHow many channels does {roi_name} have? ")
            roi_num_ch = int(roi_num_ch) - 1
            end_ch = start_ch + roi_num_ch
            roi_info = (roi_name, start_ch, end_ch)
            roi_s.append(roi_info)
            start_ch = end_ch + 1
    # check that first and last channel are included
    first_ch, last_ch = False, False
    for _, st, end in roi_s:
        if st == 0:
            first_ch = True
        if end == num_ch - 1:
            last_ch = True
    err_txt = "Channels 1 and {num_ch} not included in user setup."
    assert first_ch and last_ch, err_txt

    animals = []
    for d in dirs:
        name = "doesn't matter :)"
        if saveLFP or saveAnalog:
            name = input(f"\nWhat is the animal's ID for {d}?\n\n")
        assert name != "", "names cannot be empty"
        animals.append(name)

    processing_start = time.time()
    
    # ask for user inputs before this long loop if possible!
    dir_workers = []
    for animal_id, d in zip(animals, dirs):
        args = {
            'd': d,
            'roi_s': roi_s,
            'num_ch': num_ch,
            'saveLFP': saveLFP,
            'save_dir': save_dir,
            'animal_id': animal_id,
            'saveAnalog': saveAnalog,
            'subsample_factors': subsample_factors
        }
        # start each experiment / directory in a parallel thread
        worker = threading.Thread(target=dir_worker, kwargs=args)
        worker.start()
        dir_workers.append(worker)
    # wait for them all to finish
    for w in dir_workers:
        w.join()
            
print(f"{(time.time() - processing_start) / 60:.2f} minutes to finish processing")

if gui:
    gui_root.destroy()

# cupy / GPU memory cleanup            
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
mempool.free_all_blocks()
pinned_mempool.free_all_blocks()
