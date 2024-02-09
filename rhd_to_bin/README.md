### General Instructions

You can specify multiple input directories with RHD files. Doing this assumes that your probe setup is the same across experiments / directories.

The script will generate a <i>CRASHED_removed_at_end</i> file at the beginning, but this is for detecting silent errors that can occur from time to time. It is safe to ignore, unless you see it after the program terminates / finishes. The last step of ProcessRhd.py is to remove the CRASHED file, to signal that everything went smoothly. In addition, a log.txt file is created at the end, which currently only tells you how many RHD files were found and processed. This is another sanity check :)

### Development Tests

You should test any changes to the code that you make. You can run test_ProcessRhd.py to do this. Before running test_ProcessRhd.py you need to install [this gdrive directory](https://drive.google.com/drive/folders/1co9X7UL66yczM1iGRlUCzbRB__GyKmRH?usp=sharing). You then need to run your newest version of ProcessRhd.py on the RHDs in that directory. The image included below contains inputs that you will need to replicate exactly to ensure correct results.

![image](https://github.com/bainro/mcn_ephys/assets/31903812/3b5c0e4d-412f-4bee-8a43-2d9527a59717)
