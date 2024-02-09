import os
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
    assert os.path.exists(old_dir), f'Old results directory {old_dir} could not be found :('
    
    new_dir = input("Where are the latest results saved? These will be compared to the old gdrive results. ")
    assert os.path.exists(new_dir), f'New results directory {new_dir} could not be found :('

    # Only included one binary to save gdrive space
    files_to_compare = [
        "VC_shifted_merged.bin", 
        "a1-lfp.npy", 
        "a1-lfpts.npy", 
        "a1-digInts.npy", 
        "a1-digInts.npy", 
        "a1-analogIn.npy"
    ]

    for f in files_to_compare: 
        old = os.path.join(old_dir, f)
        new = os.path.join(new_dir, f)
        assert hash_files([old, new]), "{old} and {new} are different! :("
