import numpy as np
from array import array
from os import stat
import matplotlib.pyplot as plt
import socket
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator




# read in I/Q samples from binary file
filename = r"C:\Users\yanni\OneDrive\Documents\Hochschule\5.Semester\PAIND\loraShortShort.sig"
n_float32 = int(stat(filename).st_size/4)
with open(filename,"rb") as binary_file:
    arr = array('f')
    arr.fromfile(binary_file,n_float32)
rx = np.array(arr.tolist()[::2])+1j*np.array(arr.tolist()[1::2])
rx /= np.amax(np.abs(rx)) # normalize to 1
rx -= np.mean(rx)    # remove DC
print("signal size:",rx.shape[0])