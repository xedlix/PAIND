import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

SF = 7                      # Spreading Factor
N = 2**SF                   # Bits/Symbol
B = 125_000                 # Bandbreite
fs = B                      # Abtastfrequenz
fb = 868_000_000            # Basisbandfrequenz

# --------------------------------------------------

def analysis_DFTFB(x,w,fftlen,ts):
    # analysis FB
    # x      : input signal
    # w      : window
    # fftlen : FFT size, not exceeding the window length
    # ts     : downsampling factor, not exceeding the window length

    x_shape = x.shape
    y = np.reshape(x,(int(np.prod(x_shape[0:-1])),x_shape[-1]))
    n_symbols = int(np.ceil(x_shape[-1]/ts))
    z = np.zeros((int(np.prod(x_shape[0:-1])),n_symbols,fftlen),dtype=np.complex64)
    for i in range(y.shape[0]):
        yi = np.concatenate((y[i,:],np.zeros((ts-(x_shape[-1]%ts))%ts)))
        yi = yi.reshape((n_symbols,ts))
        wi = np.zeros((n_symbols,len(w)),dtype=np.complex64)
        for j in range(0,int(np.ceil(len(w)/ts))):
            i1 = np.minimum((j+1)*ts,len(w))
            wi[0:n_symbols-j,j*ts:i1] = np.tile(w[j*ts:i1],(n_symbols-j,1))*yi[j:,0:i1-j*ts]
        for j in range(1,int(np.ceil(len(w)/fftlen))):
            i1 = np.minimum((j+1)*fftlen,len(w))
            wi[:,0:i1-j*fftlen] = wi[:,0:i1-j*fftlen]+wi[:,j*fftlen:i1]
        z[i,:,:] = np.fft.fft(wi[:,0:fftlen])
    return z.reshape((x_shape[0:-1]+(n_symbols,fftlen)))

def display_spectrogram(x):
    fftlen = 64
    ts = 2
    w = np.sqrt(1/1536)*(1+np.cos(np.arange(-np.pi,np.pi,2*np.pi/fftlen)))/2

    z = 20*np.log10(np.abs(np.fft.fftshift(analysis_DFTFB(x,w,fftlen,ts),axes=1)))+1000
#       [Bug]: Colorbar does not work for negative values with contour/contourf #21882
    # display spectrum
    levels = MaxNLocator(nbins=15).tick_values(z.min(),z.max())
    cmap = plt.get_cmap('PiYG')
    norm = BoundaryNorm(levels,ncolors=cmap.N,clip=True)
    xp,yp = np.mgrid[slice(0,z.shape[0]),slice(0,z.shape[1])]           #ALT:    xp,yp = np.mgrid[slice(0,z.shape[0]),slice(0,z.shape[1])]

    fig = plt.figure()
    im = fig.gca().pcolormesh(xp,yp,z,cmap=cmap,norm=norm)
    fig.colorbar(im,ax=fig.gca())
    fig.gca().set_title('spectrogram')
    fig.gca()
    plt.show()

# ----------------------------------------------------

def GetUpchirp(SF):
    N = 2**SF
    n = np.arange(N, dtype=np.float32)

    x = np.exp(2*np.pi*1j*(np.square(n)/(2*N)-n/2))
    return x

def GetDownchirp(SF):
    return np.conj(GetUpchirp(SF))

def GetSymbol(SF, S, sto_frac=0, R=1):
    # SF        = Spreding Factor {7,..,12}
    # N         = bits/symbol
    # S         = Symbol
    # sto_frac  = sample time offset < 1 sample (fraction)
    # n         = number of symbols

    N = 2**SF
    n = np.arange(N * R, dtype=np.float32)/R - sto_frac

    idx_fold = (N - S)*R
    sym = np.zeros(N*R, dtype=np.complex64)
    n1, n2 = n[0:idx_fold], n[idx_fold:]
    sym[0:idx_fold]  = np.exp(2*np.pi*1j*(np.square(n1)/(2*N) + S*n1/N - n1/2))
    sym[idx_fold:] = np.exp(2*np.pi*1j*(np.square(n2)/(2*N) + S*n2/N - 3*n2/2))

    return sym

def GetPreamble(SF, net_id=64, sto_frac=0, R=1):
    # sto_frac \in [0, 1/R[
    N = 2**SF
    n = np.arange(N * R, dtype=np.float32)/R - sto_frac
    #print(n)

    upchirp = np.exp(2*np.pi*1j*(np.square(n)/(2*N)-n/2))
    downchirp = np.conj(upchirp)

    # idx_fold = (N - net_id)*R
    # sync_word = np.zeros(N*R, dtype=np.complex64)
    # n1, n2 = n[0:idx_fold], n[idx_fold:-1]
    # sync_word[0:idx_fold]  = np.exp(2*np.pi*1j*(np.square(n1)/(2*N) + net_id*n1/N - n1/2))
    # sync_word[idx_fold:-1] = np.exp(2*np.pi*1j*(np.square(n2)/(2*N) + net_id*n2/N + n2/2))
    sync_word = GetSymbol(SF, net_id, sto_frac=sto_frac, R=R)

    quarter_downchirp = downchirp[0:int(N*R/4)]
    #preamble = np.concatenate([np.tile(GetUpchirp(SF), 8), np.tile(sync_word, 2), np.tile(GetDownchirp(SF), 2), quarter_downchirp])
    preamble = np.concatenate([np.tile(GetUpchirp(SF), 8), np.tile(sync_word, 2), np.tile(GetDownchirp(SF), 2), quarter_downchirp, GetUpchirp(SF)])
    return preamble

# -----------------------------------------------------

if __name__ == "__main__":

    upchirp = GetUpchirp(SF)
    preamble = GetPreamble(SF)


    print("signal size:",upchirp.shape[0])

    display_spectrogram(upchirp)
    display_spectrogram(preamble)

    # -----------------------------------------------------

    # save in I/Q samples to a binary file
    filename = r"C:\Users\yanni\OneDrive\Documents\Hochschule\5.Semester\PAIND\preamble.sig"
    preamble.tofile(filename,sep='',format='%f')

