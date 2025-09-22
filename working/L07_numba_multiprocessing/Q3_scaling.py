import os
import numpy as np
os.environ['MKL_NUM_THREADS']      = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
np.seterr(divide='ignore', invalid='ignore')

import pycbc.psd
import pycbc.noise
import multiprocess as mp
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector

L1 = Detector("L1")

def run(M, q, ra, dec, pol, psd, duration=128, sampling_frequency=4096):   
    mass1 = M/(1+q)
    mass2 = q * mass1

    hplus, hcross = get_td_waveform(approximant="IMRPhenomXPHM", mass1=mass1, mass2=mass2, f_lower=2, delta_t=1/sampling_frequency)
        # Adjust the length of the waveform to be exactly 'duration' seconds
    if hplus.duration < duration:
        n = int((duration - h.duration) * sampling_frequency)
        hplus.prepend_zeros(n)
        hcross.prepend_zeros(n)
    elif hplus.duration > duration:
        diff = hplus.duration - duration
        hplus.crop(left=diff, right=0)
        hcross.crop(left=diff, right=0)
    
    h = L1.project_wave(hplus, hcross, ra, dec, pol)

    snr = optimal_snr(h, psd, sampling_frequency)
    
    return snr

def noise_weighted_inner_product(a, b, psd, duration):
    """
    Computes the noise weighte inner product of two frequency domain signals a and b.
    
    Args:
        a   (numpy.ndarray): frequency domain signal
        b   (numpy.ndarray): frequency domain signal
        psd (numpy.ndarray): power spectral density
        duration    (float): duration of the signal
    
    """
    integrand = np.conj(a) * b / psd
    return (4 / duration) * np.sum(integrand, -1)


def optimal_snr(td_template, psd, sampling_frequency):
    """
    Computes the optimal SNR of a signal.
    The code is adapted from Bilby 
    (https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/gw/utils.py?ref_type=heads)
    
    Args:

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

    # set seed
    import random
    random.seed(42)
    np.random.seed(42)

    N = 10
    M = np.random.uniform(100, 1000, N)
    q = np.random.uniform(0.125, 1, N)
    ra  = np.random.uniform(0, 2*np.pi, N)
    dec = np.arcsin(np.random.uniform(-1, 1, N))
    pol = np.random.uniform(0, np.pi, N)

    # The color of the noise matches a PSD which you provide
    flow = 2
    delta_f = 1.0 / 128
    flen = int(4096 / delta_f) + 1
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

    # Generate 32 seconds of noise at 4096 Hz
    delta_t = 1.0 / 4096
    tsamples = int(128 / delta_t)
    noise = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=42)

    run_times = []
    cpus_test = range(1, mp.cpu_count()+9)
    for cpus in tqdm(cpus_test):
        print(f"Running with {cpus} CPUs")
        mp.set_start_method('fork', force=True)
        start = time()

        with mp.Pool(processes=cpus) as pool:
            results = list(pool.imap_unordered(lambda x: run(x[0], x[1], x[2], x[3], x[4], x[5]), 
                                        zip(M, q, ra, dec, pol, psd)))
        end = time()
        run_times.append(end - start)
        
    plt.figure()
    plt.plot(cpus_test, run_times, marker='o')
    plt.xlabel("Number of CPUs")
    plt.ylabel("Run time [s]")
    plt.axvline(os.cpu_count(), color='k', ls='--', label='Number of physical cores')
    plt.title("Scaling of SNR computation with number of CPUs")
    plt.legend()
    plt.savefig("scaling.png", dpi=200, bbox_inches='tight')