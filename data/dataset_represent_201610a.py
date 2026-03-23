# !pip install pickle5
# !pip install numpy

import pickle
import numpy as np

# dataset path
dataset_path = "RML2016.10a_dict.pkl"

# load dataset
with open(dataset_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# modulation types and SNR levels
mods = sorted(list(set([k[0] for k in data.keys()])))
snrs = sorted(list(set([k[1] for k in data.keys()])))

print("Modulations:", mods)
print("SNR levels:", snrs)

# build dataset
X = []
Y = []

for mod in mods:
    for snr in snrs:
        signals = data[(mod, snr)]
        for sig in signals:
            X.append(sig)
            Y.append(mod)

X = np.array(X)
Y = np.array(Y)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Example signal shape:", X[0].shape)