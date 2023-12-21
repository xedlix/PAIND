# Importig data
from os import stat
from array import array

# Plotting and grafics
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# Math
import numpy as np

# Selfmade moduls 
import Preambel as pre


DEBUG_INFOS = True

# Communication Parameters
BANDWIDTH_HZ = 125_000
SPREADING_FACTOR = 7

PREAMBLE_LENGTH = 8
DOWNCHIRP_LENGHT = 2
NETWORK_IDENTIFIER_LENGHT = 2

# Positions
LAST_DOWN_CHIRP_POSITION = PREAMBLE_LENGTH - 1
FIRST_UP_CHIRP_POSITION = PREAMBLE_LENGTH + 2

#Symbols
STEP_SIZE = 2**SPREADING_FACTOR
NORMALIZED_OFFSET = -0.5
SIGNAL_SIZE = 55    # number of symbols in a signal

#Data Extraction
FIRST_DATA_POINT = PREAMBLE_LENGTH + NETWORK_IDENTIFIER_LENGHT + DOWNCHIRP_LENGHT
SYNC_OFFSET = int(STEP_SIZE/4)

# Loading data from File------------------------------------------------------------------------------------

def load_signal(filename:str)->complex:
    """
    loads a signal from a file generatete my the GNU-Radio Compannion

    Parameters:
        filename    (str)           : Relative or absolute path to the file
    
    Returns:
        signal_rx   (complex array) : Signal in timeform
    """
    # filename = r"Signale\lora16x3.sig"
    n_float32 = int(stat(filename).st_size/4)
    with open(filename,"rb") as binary_file:
        arr = array('f')
        arr.fromfile(binary_file,n_float32)
    signal_rx = np.array(arr.tolist()[::2])+1j*np.array(arr.tolist()[1::2])
    signal_rx /= np.amax(np.abs(signal_rx)) # normalize to 1
    signal_rx -= np.mean(signal_rx)    # remove DC

    if DEBUG_INFOS:
        print(f"signal size: {signal_rx.shape[0]}")

    return signal_rx


# Spectrum & Symbol visualization---------------------------------------------------------------------------

def analysis_DFTFB(signal_t:complex,w:int,fftlen:int,ts:int)->complex:
    """ 
    Discrete Fourier Transformation with Filter Banks.

    CREDITS: This Function it copied from Thomas Hunziker @HSLU T&A

    Parameters: 
        signal_t    (complex array) : Input signal in timeform
        w           (int)           : window
        fftlen      (int)           : FFT size, not exceeding the window length
        ts          (int)           : Downsampling factor, not exceeding the window length

    Returns:
        -
    """
    x_shape = signal_t.shape
    y = np.reshape(signal_t,(int(np.prod(x_shape[0:-1])),x_shape[-1]))
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

def display_spectrogram(signal_t:complex)->None:
    """ 
    Prints the spectrum chunkwise of 'signal_t'. 

    CREDITS: This Function it copied from Thomas Hunziker @HSLU T&A

    Parameters: 
        signal_t        (complex array) : Input signal in timeform

    Returns:
        -
    """
    fftlen = 128
    ts = 4
    w = np.sqrt(1/1536)*(1+np.cos(np.arange(-np.pi,np.pi,2*np.pi/fftlen)))/2

    z = 20*np.log10(np.abs(np.fft.fftshift(analysis_DFTFB(signal_t,w,fftlen,ts),axes=1)))+1000
#       [Bug]: Colorbar does not work for negative values with contour/contourf #21882
    # display spectrum
    levels = MaxNLocator(nbins=15).tick_values(z.min(),z.max())
    cmap = plt.get_cmap('PiYG')
    norm = BoundaryNorm(levels,ncolors=cmap.N,clip=True)
    xp,yp = np.mgrid[slice(0,z.shape[0]),slice(0,z.shape[1])]
    fig = plt.figure()
    im = fig.gca().pcolormesh(xp,yp,z,cmap=cmap,norm=norm)
    fig.colorbar(im,ax=fig.gca())
    fig.gca().set_title('spectrogram')
    plt.show()

def display_spectrogram_if_signal(signal_t:complex,signal_strength:int)->None:
    """ 
    Prints the spectrum chunkwise of 'signal_t', but only if there hat at least one datapoint
    a higher strenght than the set value 'signal_strenght'. This is much more 
    timeefficent than print all.

    CREDITS: This Function it copied from Thomas Hunziker @HSLU T&A

    Parameters: 
        signal_t        (complex array) : Input signal in timeform
        signal_strength (int)           : trigger value to print the spectrum of this sampleblock

    Returns:
        -
    """
    fftlen = 128
    ts = 4
    w = np.sqrt(1/1536)*(1+np.cos(np.arange(-np.pi,np.pi,2*np.pi/fftlen)))/2

    z = 20*np.log10(np.abs(np.fft.fftshift(analysis_DFTFB(signal_t,w,fftlen,ts),axes=1)))+1000
#       [Bug]: Colorbar does not work for negative values with contour/contourf #21882
    if(z.max() >= signal_strength):
        # display spectrum
        levels = MaxNLocator(nbins=15).tick_values(z.min(),z.max())
        cmap = plt.get_cmap('PiYG')
        norm = BoundaryNorm(levels,ncolors=cmap.N,clip=True)
        xp,yp = np.mgrid[slice(0,z.shape[0]),slice(0,z.shape[1])]
        fig = plt.figure()
        im = fig.gca().pcolormesh(xp,yp,z,cmap=cmap,norm=norm)
        fig.colorbar(im,ax=fig.gca())
        fig.gca().set_title('spectrogram')
        plt.show()

def visualize_symbols(signal_t:complex, startpoint:int = None, title:str = "Demoduliertes Signal") -> None:
    """ 
    Prints a Graph, where you can see the index and the value at the index.

    Parameters: 
        signal_t    (complex array) : input signal in timeform
        title       (string)        : the title of the graph

    Returns:
        -
    """

    index,symbols = demodulate_symbols(signal_t)
    plt.figure().set_figwidth(20)
    plt.plot(index,symbols,'o',)
    if startpoint is not None:
        plt.plot(index[startpoint+LAST_DOWN_CHIRP_POSITION],symbols[startpoint+LAST_DOWN_CHIRP_POSITION],'s',)
        plt.plot(index[startpoint+FIRST_UP_CHIRP_POSITION],symbols[startpoint+FIRST_UP_CHIRP_POSITION],'s',)

    plt.ylim([1, 2**SPREADING_FACTOR*1.25])

    
    plt.xlabel("Samples")
    plt.ylabel("Max Symbol") # TODO: Besserer Name
    plt.title(title)
    # X => Offset vom Symbol => Symbolwert => ACHTUNG: Beinhaltet CFO & STO")
   
    steps_x = np.arange(0,len(signal_t),(2**SPREADING_FACTOR))
    steps_y = np.arange(0,144,16)

    plt.yticks(steps_y)
    plt.xticks(steps_x)
    plt.grid()

    # labeling of the X & Y coordinats of each point 
    for i in range(len(index)):
        plt.annotate(f"Y{symbols[i]}", (index[i], symbols[i]), xytext=(index[i], symbols[i] + 5))
        plt.annotate(f"X{index[i]-(i*(2**SPREADING_FACTOR))}", (index[i], symbols[i]), xytext=(index[i], symbols[i] + 10))           # shows the value of the Y value

    plt.show()


# Downsampling & Signal cutting-----------------------------------------------------------------------

def downsample_signal(signal_t:complex,factor:int) -> complex:
    return signal_t[::factor]

def cut_signal(signal_t:complex) -> complex:
    """
    It cuts the inputfile down to the signal with a fixed lenght

    Parameters:
        signal_t        (complex array) : Input signal in timeform

    Returns:
        signal_cut_t    (complex array) : Input signal cut to the signal length
    """
    index, symbols, start_position = demodulate_detected(signal_t)

    start_index = (start_position - 1) * STEP_SIZE
    stop_index = (start_position + SIGNAL_SIZE) * STEP_SIZE

    signal_cut_t = signal_t[start_index:stop_index]
    return signal_cut_t


# Preamble detection-----------------------------------------------------------------------

def preamble_to_delta(index:complex,symbols:complex) -> tuple[complex,complex]:
    """ 
    subtracts the index to the offset 

    Parameters: 
        index   (complex array) : Position of each frequency-jump along all samples 
        symbols (complex array) : Maximum Value of each frequency-jump

    Returns:
        index   (complex array) : Offset of each frequency-jump to a fixed symbollenght of 128 Samples 
        symbols (complex array) : Maximum Value of each frequency-jump (unchanged)
    """

    index_delta = index.copy()
    symbols_delta = symbols
    for x in range(len(index)):
        index_delta[x] =index[x] - 128*x
    return(index_delta,symbols_delta)

def preamble_detection(index:complex,symbols:complex) -> int:
    """ 
    detects a preamble in the date if ther is one and returns the position
    Parameters: 
        index       (complex array) : Position of each frequency-jump along all samples 
        symbols     (complex array) : Maximum Value of each frequency-jump

    Returns:
        position    (int)           : Gives the startpoint of the preample in the index array back. If none -1 will be returned
    """
        
    index_delta,symbols_delta = preamble_to_delta(index,symbols)

    position = 0
    number_of_detected_symbols = 0

    # As long as there is signal
    while (position + 1) < len(index_delta):

        # Index in bound of +-1
        if(index_delta[position+1] in range(index_delta[position-number_of_detected_symbols]-1,index_delta[position-number_of_detected_symbols]+1)):
            number_of_detected_symbols += 1
            position += 1

        # Index out of bound of +-1
        else:
            number_of_detected_symbols = 0
            position += 1

        # positiv endcase
        if number_of_detected_symbols >= (PREAMBLE_LENGTH - 1):
            if DEBUG_INFOS:
                print(f"Startposition of the preamble: {position - PREAMBLE_LENGTH + 1 }")
            return (position - PREAMBLE_LENGTH + 1 )
        
    return -1  # no signal found

def get_start_position(signal_t:complex)->int:
    """
    searches for the pramble sequence and get its position in the index array

    Parameters:
        signal_t            (complex array) : input signal in timeform

    Returns:
        start_position       (int)          : position of the first preamble symbol
    
    """

    index,symbols = demodulate_symbols(signal_t)
    start_position = preamble_detection(index,symbols)
    if DEBUG_INFOS:
        print(f"Start of the Pramble: {start_position}")
    return start_position


# Demodulation------------------------------------------------------------------------

def demodulate_symbols(signal_t:complex):
    """ 
    This function is used to demodulate the symbols out of the signal in the timeform.
    The output is the index array witch contains the uncorrected symbols with the delta.
    This functions uses the "demoudulate_symbols_helper"

    Parameters:
        signal_t    (complex array) : input signal in timeform

    Returns:
        index       (complex array) : position of each frequency-jump along all samples
        symbols     (complex array) : maximum value of each frequency-jump (unused)
    """

    (index,symbols) = demodulate_symbols_helper(signal_t)
    start_position_of_preamble = preamble_detection(index,symbols)
    (index,symbols) = demodulate_symbols_helper(signal_t,start_position_of_preamble)
    return (index,symbols)

# Private
def demodulate_symbols_helper(signal_t:complex, start_position_of_preamble:int = -1) -> tuple[complex,complex]:
    """ 
    This is a helperoutine to demodulate all symbols. Its is local function without acsess from outside

    Parameters:
        signal_t                    (complex array) : Input signal in timeform
        start_position_of_preamble  (int)           : Position of the first downchirp of the pramble in symbols in signal_t

    Returns:
        index                       (complex array) : Position of each frequency-jump along all samples 
        symbols                     (complex array) : Maximum value of each frequency-jump (unused)
    """
    signal_zero_up = pre.GetUpchirp(SPREADING_FACTOR)               # referencesignal
    STEP = len(signal_zero_up)                                      # size of referencesignal

    if DEBUG_INFOS and False:
        print(f"demodulate symbols:")
        print(f"    size of signal_t    : {len(signal_t)}")
        print(f"    size of symbol zero : {STEP} \n")


    if len(signal_t) < STEP:
        raise IndexError("the input signal is shorter than a step") # Error: signal is to short

    position = 0
    symbols = []
    index = []
    
    while len(signal_t) >= STEP:

        if ((position - start_position_of_preamble ==  10) or (position - start_position_of_preamble == 11 )) and start_position_of_preamble >= 0:
            signal_td = np.multiply(signal_t[0:STEP],signal_zero_up)
        else:
            signal_td = np.multiply(signal_t[0:STEP],np.conj(signal_zero_up))

        signal_f = abs(np.fft.fft(signal_td))
        symbols.append(round(max(signal_f)))            #findes always the highest value
        index.append(position*STEP + np.array(signal_f).argmax())
        signal_t = signal_t[STEP::]                     # shorten the array by a step
        position += 1
                                   
    return (index,symbols)


# Combinerfunctions-----------------------------------------------------------------------------------------------

def demodulate_detected(signal_t:complex)-> tuple[complex,complex,int]:
    """ 
    Combines the functions of "demodulate_symbols" and "preamble_detection"

    Parameters:
        signal_t                    (complex array) : input signal in timeform

    Returns:
        index                       (complex array) : position of each frequency-jump along all samples
        symbols                     (complex array) : maximum value of each frequency-jump (unused)
        start_position_of_preamble  (int)           : position of the first downchirp of the pramble in symbols in signal_t
    """

    index,symbols = demodulate_symbols(signal_t)
    start_position = preamble_detection(index,symbols)
    return (index,symbols,start_position)

def demodulate_detected_corrected(signal_t:complex)->complex:
    """ 
    Combines the functions of "demodulate_symbols", "preamble_detection" and "correction_int"

    Parameters:
        signal_t            (complex array) : input signal in timeform

    Returns:
        signal_corrected    (complex array) : CFO & STO corrected signal
    """
   
    start_position_of_preamble = get_start_position(signal_t)

    # Errorhandling
    if start_position_of_preamble < 0:
        raise IndexError("Index has to be positiv => demodualation failed")
    
    signal_corrected = correction_int(signal_t)
    visualize_symbols(signal_corrected,start_position_of_preamble,"signal_corrected") # visualise
    return signal_corrected


# CFO & STO correction--------------------------------------------------------------------------------

def correction_int(signal_t:complex)->complex:
    """ 
    corrects the inputsignal agianst CFO & STO

    Parameters:
        signal_t                (complex array) : input signal in timeform

    Returns:
        signal_int_corrected    (complex array) : CFO & STO corrected signal
    """
    
    # demodulate signal
    (index_delta, symbols_delta, start_position_of_preamble) = demodulate_detected(signal_t)

    # get the normalised frequencies
    frequency_upchirp_normalized = index_delta[start_position_of_preamble + LAST_DOWN_CHIRP_POSITION ] / STEP_SIZE + NORMALIZED_OFFSET  # Last frequency of the upchirp normalised
    frequency_downchirp_normalized = index_delta[start_position_of_preamble + FIRST_UP_CHIRP_POSITION] / STEP_SIZE + NORMALIZED_OFFSET  # first frequency of the downchirp normalised
    
    # CFO ---------------------
    # CFO offset calculations
    if(frequency_upchirp_normalized >= -frequency_downchirp_normalized):
        center_frequency_offset_int = round((frequency_upchirp_normalized + frequency_downchirp_normalized ) / 2 * BANDWIDTH_HZ)
    else:
        center_frequency_offset_int = round((frequency_upchirp_normalized + frequency_downchirp_normalized ) / 2 * BANDWIDTH_HZ + BANDWIDTH_HZ/2)

    # CFO correction
    center_frequency_shift_int = np.exp(2j*np.pi*np.arange(len(signal_t))*-center_frequency_offset_int/BANDWIDTH_HZ)
    signal_cfo_int_corrected = np.multiply(signal_t,center_frequency_shift_int)
    
    # STO ---------------------
    # STO offset calculations
    if(frequency_upchirp_normalized >= frequency_downchirp_normalized):
        sample_time_offset_int = round((frequency_upchirp_normalized - frequency_downchirp_normalized) / 2 * STEP_SIZE)
    else:
        sample_time_offset_int = round((frequency_upchirp_normalized - frequency_downchirp_normalized) / 2 * STEP_SIZE + STEP_SIZE/2)     
  
    if sample_time_offset_int < 0:
        sample_time_offset_int = STEP_SIZE + sample_time_offset_int

    # STO correction
    signal_int_corrected = np.concatenate((np.zeros(sample_time_offset_int,dtype=complex),signal_cfo_int_corrected),dtype=complex)  # add zeros to the start of the preamble

    #  DEBUG OUTPUTS ------------
    if DEBUG_INFOS and False:
        print(f"start position of the preamble: {start_position_of_preamble}")
        visualize_symbols(signal_cfo_int_corrected,start_position_of_preamble,"signal_cfo_int_corrected")
        print(f"correction of the Carrier Frequency Offset:")
        print(f"    CFO: {round(center_frequency_offset_int / BANDWIDTH_HZ *STEP_SIZE)} \n")
        print(f"correction of the Sample Time Offset:")
        print(f"    STO: {sample_time_offset_int} \n")
        visualize_symbols(signal_int_corrected,start_position_of_preamble,"signal_int_corrected")

    return signal_int_corrected


# Data extractor------------------------------------------------------------------------------

def corrected_to_data(signal_t:complex)->int:
    """
    takes the int_corrected signal and converts it to the raw data without the preamble

    Parameters:
        signal_t    (complex array) : input signal in timeform

    Returns:
        data        (int array)    : raw data of the signal_t
    
    """
    # Cut preamble away and correct 1/4 sync Symbol
    start_index = FIRST_DATA_POINT * STEP_SIZE + SYNC_OFFSET
    stop_index = (FIRST_DATA_POINT + SIGNAL_SIZE) * STEP_SIZE + SYNC_OFFSET
    signal_cut_t = signal_t[start_index:stop_index]

    index,symbols,start_position = demodulate_detected(signal_cut_t)
    index_delta,symbols_delta = preamble_to_delta(index,symbols)
    return index_delta


# MAIN FUNCTION------------------------------------------------------------------------------

def signal_to_data(signal_rx:complex, downsampling_faktor:int = 16)->int:
    """
    takes the signal befor Downsamling and gives back the raw data

    Parameters:
        signal_rx           (complex array) : Input signal in timeform 
        downsampling_faktor (int)           : Downsampling faktor to correct the signal

    Returns:
        data                (int)           : Raw data of the signal

    """
    # Downsampling
    signal_t = downsample_signal(signal_rx,downsampling_faktor)

    # detect preamble and cut signal out
    signal_t = cut_signal(signal_t)

    # offset corrections
    signal_t = correction_int(signal_t)

    # cut pramble away and correct for frame sync
    data = corrected_to_data(signal_t)

    # return the raw data
    return data

