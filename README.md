# Installation:
```
git clone https://github.com/bainro/mcn_ephys.git 
conda create -n ephys -y cython h5py joblib matplotlib pillow pip requests responses scikit-learn scipy spyder opencv conda-forge::cupy python=3.9
conda activate ephys
```

## Run with spyder:
```
cd mcn_ephys
spyder -p .
```

## Run on the command line:
```
cd mcn_ephys
python rhd_to_bin/ProcessRhd.py
```
