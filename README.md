## Installation:
Requires [anaconda](https://www.anaconda.com/download), [git](https://git-scm.com/download/win](https://git-scm.com/downloads), and nvidia GPU drivers (can check it by running ```nvidia-smi``` on the command line).
```
git clone https://github.com/bainro/mcn_ephys.git
cd mcn_ephys
conda create -n ephys -y cython h5py joblib matplotlib pillow pip requests responses scikit-learn scipy spyder opencv natsort conda-forge::cupy python=3.9
conda activate ephys
```

### Download the latest code (i.e. update already downloaded code)
```
cd mcn_ephys
git pull
```

### If you want to try out the development branch (more bugs guaranteed!):
```git checkout dev```

### Switch back to the release/main branch:
```git checkout main```

### Run with spyder:
```
spyder -p .
```

### Or run on the command line:
```
python rhd_to_bin/ProcessRhd.py
```

#### Contributions:
Original [repo](https://github.com/rajatsaxena/mea/tree/main) of Rajat Saxena's.
