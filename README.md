# Installation:
```
git clone https://github.com/bainro/mcn_ephys.git
cd mcn_ephys
conda create -n ephys -y cython h5py joblib matplotlib pillow pip requests responses scikit-learn scipy spyder opencv conda-forge::cupy python=3.9
conda activate ephys
```

## Download the latest code (i.e. update already downloaded code)
```
cd mcn_ephys
git checkout main
git pull
```

## Try out the development branch (more bugs guaranteed!):
```git checkout dev```

## Switch back to the release/main branch:
```git checkout main```

## Run with spyder:
```
spyder -p .
```

## Run on the command line:
```
python rhd_to_bin/ProcessRhd.py
```
