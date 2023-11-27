import numpy as np
from array import array
from os import stat
import matplotlib.pyplot as plt
import socket
import scipy.signal as signal

# Inputs
F_SAMPLE = 150000           # Sample Rate
SF = 7                      # Spreading Factor
BW = 125e3                  # Bandbreite der Übertragung

online = 0

n = 2^SF                  # bits pro Symbol
f_Hz = F_SAMPLE*(np.arange(8192)/8192-0.5) #????

if online:
    # read in I/Q samples from binary file
    filename = "..\lora.sig"
    n_float32 = int(stat(filename).st_size/4)
    with open(filename,"rb") as binary_file:
        arr = array('f')
        arr.fromfile(binary_file,n_float32)
    rx = np.array(arr.tolist()[::2])+1j*np.array(arr.tolist()[1::2])
    rx /= np.amax(np.abs(rx))
    rx -= np.mean(rx)    # remove DC
    print("signal size:",rx.shape[0])
else:
    # socket (server) read loop
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.bind(("",52001))
    s.listen(1)
    conn,addr = s.accept()
    data = bytearray([])
    rx = np.zeros(0,dtype=np.complex64)

while True:
    while (len(data)<65536):
        data = data+conn.recv(16384)
    rx = np.frombuffer(data[0:65536],dtype=np.complex64)
    data = data[65536:len(data)]


    # Downsamling -> Überlagerung aller möglicher Frequenzbänder
    rxd = signal.decimate(data,8)

    # display spectrum
    f_dB = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(rx*np.sin(np.arange(8192)*np.pi/8192)))))

    plt.xlabel('$f$    [Hz]')
    plt.plot(f_Hz,f_dB)
    plt.show()
conn.close()