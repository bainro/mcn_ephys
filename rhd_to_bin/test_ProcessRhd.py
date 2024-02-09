import os
import numpy as np
from compare_hashes import hash_files

if __name__ == "__main__":  
    print()
    print("You need to install this directory: https://drive.google.com/drive/folders/1co9X7UL66yczM1iGRlUCzbRB__GyKmRH?usp=sharing")
    rdy = input("Then run your latest ProcessRhd.py on it. Have you done this and are ready to continue? [y/n] ")
    rdy = ('y' in rdy) or ('Y' in rdy)
    if not rdy:
        print("Byeeeee!")
        exit()
        
    old_dir = input("Where did you download the gdrive folder too (full path)? ")
    
    new_txt = input("Where are the latest results saved? These will be compared to the old gdrive results. ")
    
    new_dir = input(dirs_txt).replace(", ", ",").split(',')
    assert len(new_dir) == 1, "No input directories specified :("
    assert os.path.exists(new_dir), f'New results directory {new_dir} could not be found :('

    lfp_filename = os.path.join(sub_save_dir, animal_id+'-lfp.npy')
    lfpts_filename = os.path.join(sub_save_dir, animal_id+'-lfpts.npy')
    digIn_filename = os.path.join(sub_save_dir, animal_id+'-digIn.npy')
    digIn_ts_filename = os.path.join(sub_save_dir, animal_id+'-digInts.npy')
    analogIn_filename = os.path.join(sub_save_dir, animal_id+'-analogIn.npy') 
