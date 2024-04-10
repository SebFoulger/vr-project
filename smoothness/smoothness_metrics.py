
"""
smoothness.py contains a list of functions for estimating movement smoothness.
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def sparc(movement, fs, padlevel=4, fc=10.0, amp_th=0.05, return_spectrum: bool = False):
    """
    Calcualtes the smoothness of the given speed profile using the modified
    spectral arc length metric.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    padlevel : integer, optional
               Indicates the amount of zero padding to be done to the movement
               data for estimating the spectral arc length. [default = 4]
    fc       : float, optional
               The max. cut off frequency for calculating the spectral arc
               length metric. [default = 10.]
    amp_th   : float, optional
               The amplitude threshold to used for determing the cut off
               frequency upto which the spectral arc length is to be estimated.
               [default = 0.05]
    return_spectrum : bool, optional
               Indicate whether to return the spectrum as well as the calcualted value
               
    Returns
    -------
    sal      : float
               The spectral arc length estimate of the given movement's
               smoothness.
    (f, Mf)  : tuple of two np.arrays
               This is the frequency(f) and the magntiude spectrum(Mf) of the
               given movement data. This spectral is from 0. to fs/2.
    (f_sel, Mf_sel) : tuple of two np.arrays
                      This is the portion of the spectrum that is selected for
                      calculating the spectral arc length.

    Notes
    -----
    This is the modfieid spectral arc length metric, which has been tested only
    for discrete movements.

    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> sal, _, _ = sparc(move, fs=100.)
    >>> '%.5f' % sal
    '-1.41403'

    """
    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

    # Frequency
    f = np.arange(0, fs, fs / nfft)
    # Normalized magnitude spectrum
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf / max(Mf)

    # Indices to choose only the spectrum within the given cut off frequency
    # Fc.
    # NOTE: This is a low pass filtering operation to get rid of high frequency
    # noise from affecting the next step (amplitude threshold based cut off for
    # arc length calculation).
    fc_inx = ((f <= fc) * 1).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Choose the amplitude threshold based cut off frequency.
    # Index of the last point on the magnitude spectrum that is greater than
    # or equal to the amplitude threshold.
    inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1] + 1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate arc length
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    if return_spectrum:
        return new_sal, (f, Mf), (f_sel, Mf_sel)
    else:
        return new_sal

def ldj_adl(movement, fs, movement_peak = 0, data_type='speed'):
    
    # first enforce data into an numpy array.
    movement = np.array(movement)

    dt = 1. / fs
    movement_dur = len(movement) * dt
    
    if data_type=='speed':
        movement_peak = max(abs(movement))   
        movement = np.diff(movement)
          
    scale = pow(movement_dur, 3) / pow(movement_peak, 2)

    # estimate dj
    return - np.log(scale * sum(pow(movement, 2)))

def dj(movement, fs, movement_peak=0, data_type='speed'):
    """
    Calculates the smoothness metric for the given movement data using the
    dimensionless jerk metric. The input movement data can be 'speed',
    'accleration' or 'jerk'.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    data_type: string
               The type of movement data provided. This will determine the
               scaling factor to be used. There are only three possibiliies,
               {'speed', 'accl', 'jerk'}

    Returns
    -------
    dl       : float
               The dimensionless jerk estimate of the given movement's
               smoothness.

    Notes
    -----


    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> dl = dimensionless_jerk(move, fs=100.)
    >>> '%.5f' % dl
    '-335.74684'

    """
    # first ensure the movement type is valid.
    if data_type in ('speed', 'accl', 'jerk'):
        # first enforce data into an numpy array.
        movement = np.array(movement)

        dt = 1. / fs
        movement_dur = len(movement) * dt
        
        if data_type == 'speed':
            movement_peak = max(abs(movement))
            
        scale = pow(movement_dur, 3) / pow(movement_peak, 2)

        # estimate jerk
        if data_type == 'speed':
            jerk = np.diff(movement, 2) / pow(dt, 2)
        elif data_type == 'accl':
            jerk = np.diff(movement, 1) / pow(dt, 1)
        else:
            jerk = movement

        # estimate dj
        return - scale * sum(pow(jerk, 2)) * dt
    else:
        raise ValueError('\n'.join(("The argument data_type must be either",
                                    "'speed', 'accl' or 'jerk'.")))


def ldj(movement, fs, movement_peak=0, data_type='speed'):
    """
    Calculates the smoothness metric for the given movement data using the
    log dimensionless jerk metric. The input movement data can be 'speed',
    'accleration' or 'jerk'.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    data_type: string
               The type of movement data provided. This will determine the
               scaling factor to be used. There are only three possibiliies,
               {'speed', 'accl', 'jerk'}

    Returns
    -------
    ldl      : float
               The log dimensionless jerk estimate of the given movement's
               smoothness.

    """
    return -np.log(abs(dj(movement, fs, movement_peak, data_type)))

def nop(movement, fs):
    """Calculates number of peaks per meter with a prominence at least 0.05m/s

    Args:
        movement: speed profile
        fs (int): frequency
    """    
    distance = sum(movement)/fs
    no_peaks = len(find_peaks(movement, prominence=0.05)[0])

    return -no_peaks/distance

def nos(movement, fs, breakpoints):
    prev_break=0
    count=0
    for _break in breakpoints:
        if _break - prev_break >=2:
            count+=1
        prev_break = _break
    distance = sum(movement)/fs

    return -count/distance

def nosp(movement, fs, breakpoints, betas):
    prev_break=breakpoints[0]
    count=0
    index=1

    for _break in breakpoints[1:-1]:
        if _break - prev_break >= 9:
            #print(breakpoints.index(_break), _break, index)
            if betas[index]<0 and betas[index-1]>0:
                count+=1
            index+=1
            
        prev_break = _break
    distance = sum(movement)/fs

    return -count/distance

class SegmentMetric:
    def __init__(self, metric):
        self.metric = metric

    def value(self, movement, fs: float, breakpoints: list, data_type: str = None, speed = []):
        assert(breakpoints[-1] <= len(movement))

        prev_break = 0

        metric_list = []
        weights = []

        for _break in breakpoints:

            cur_movement = movement[prev_break:_break - 1].reset_index(drop=True)
            if len(cur_movement)<10:
                continue
            if data_type is None:
                metric_list.append(self.metric(cur_movement, fs))
                
            else:
                cur_speed = speed[prev_break:_break - 1].reset_index(drop=True)
                metric_list.append(self.metric(cur_movement, fs, data_type=data_type, movement_peak = max(cur_speed)))
            weights.append(_break - prev_break)
            prev_break = _break

        metric_list = np.array(metric_list)
        weights = np.array(weights)
        weights = weights[~np.isnan(metric_list)]
        metric_list = metric_list[~np.isnan(metric_list)]

        return np.average(metric_list, weights=weights)