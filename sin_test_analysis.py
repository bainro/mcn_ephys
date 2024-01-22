import scipy
import matplotlib.pyplot as plt
import numpy as np

intan_data = scipy.io.loadmat(r'./1kHz_signal.mat')
print(intan_data.keys())

# %%

# get 1st 100 values from 1st 32 channels
channels = []
for i in range(64):
    ch_id = str(i).rjust(3, '0')
    ch_id = "A_" + ch_id
    # [channel id] [signal, time]
    ch = intan_data[ch_id][300:330,0]
    channels.append(list(ch))

# %%

plt.figure(figsize=(10,5)) 
plt.plot(channels[32], label=0)
plt.plot(channels[63], label=31)
plt.legend()

# %%

plt.figure(figsize=(50,5)) 
## Parameters used
StopTime = 500 # End of signal 
Fs = 1024      # Sampling rate
f = 2         # Frequency of simulated signal

## Generate sample times
t = np.linspace(0, StopTime, int(StopTime*Fs))

## Generate signal
x = np.sin(2*np.pi*t*f)

## Add noise to signal
noise = np.random.randn(len(x))
xn = x + noise / 20

f2 = 20         # Frequency of simulated signal
## Generate signal
x2 = np.sin(2*np.pi*t*f2)

## Add noise to signal
noise2 = np.random.randn(len(x2))
xn2 = x2 + noise2

x_combo = xn + xn2

plt.plot(t, x_combo / 50)
plt.show()

# %%

'''
x_combo = np.random.randn(len(x2))
plt.plot(x_combo)
plt.show()
'''

# %%
 
x_recon = np.real(np.fft.ifft(np.fft.fft(x_combo)))
plt.plot(t, x_recon)
plt.show()

# %%

print(sum(x_recon) - sum(x_combo))
