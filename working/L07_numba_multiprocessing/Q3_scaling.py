import os
import numpy as np
os.environ['MKL_NUM_THREADS']      = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from tqdm import tqdm
from pycbc.waveform import get_td_waveform

def generate_waveforms(N=10, m_min=100, m_max=1000, sampling_frequency=4096, duration=3600):   
    M = np.random.uniform(m_min, m_max, N)
    q = np.random.uniform(0.125, 1, N)

    m1 = M/(1+q)
    m2 = q * m1

    td_templates = []
    for mass1, mass2 in tqdm(zip(m1, m2), total=N):
        h, _ = get_td_waveform(approximant="IMRPhenomXPHM", mass1=mass1, mass2=mass2, f_lower=2, delta_t=1/sampling_frequency)
        
        if h.duration < duration:
            n = int((duration - h.duration) * sampling_frequency)
            h.prepend_zeros(n)
        elif h.duration > duration:
            diff = h.duration - duration
            h.crop(left=diff, right=0)
        td_templates.append(h)
    return np.array(td_templates)


def noise_weighted_inner_product(a, b, psd, duration):
    """
    Computes the noise weighte inner product of two frequency domain signals a and b.
    
    Args:
    -----
        a   (numpy.ndarray): frequency domain signal
        b   (numpy.ndarray): frequency domain signal
        psd (numpy.ndarray): power spectral density
        duration    (float): duration of the signal
    
    """
    integrand = np.conj(a) * b / psd
    return (4 / duration) * np.sum(integrand, dim = -1)


def optimal_snr(td_template, psd, sampling_frequency):
    """
    Computes the optimal SNR of a signal.
    The code is adapted from Bilby 
    (https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/gw/utils.py?ref_type=heads)
    
    Args:
    -----
        td_domain_template (numpy.ndarray): time domain signal
        psd                (numpy.ndarray): power spectral density
        sampling_frequency         (float): sampling frequency of the signal
    """
    duration    = len(td_template) / sampling_frequency
    fd_template = np.fft.rfft(td_template) / sampling_frequency 
    rho_opt = noise_weighted_inner_product(fd_template, 
                                           fd_template, 
                                           psd, duration)
    snr_square = np.abs(rho_opt)
    return np.sqrt(snr_square)



if __name__=="__main__":    
    h = generate_waveforms()

