import os
import math
import glob
import time
import numpy as np

if __name__ == "__main__":  
    print()
    print("You need to install this directory: ____")
    print("Where did you install it? ")
    print("Where are the latest results saved? These will be compared to the old gdrive results.")
    
    new_dir = input(dirs_txt).replace(", ", ",").split(',')
    assert len(new_dir) == 1, "No input directories specified :("
    assert os.path.exists(new_dir), f'New results directory {new_dir} could not be found :('

    lfp_filename = os.path.join(sub_save_dir, animal_id+'-lfp.npy')
    lfpts_filename = os.path.join(sub_save_dir, animal_id+'-lfpts.npy')
    digIn_filename = os.path.join(sub_save_dir, animal_id+'-digIn.npy')
    digIn_ts_filename = os.path.join(sub_save_dir, animal_id+'-digInts.npy')
    analogIn_filename = os.path.join(sub_save_dir, animal_id+'-analogIn.npy') 
