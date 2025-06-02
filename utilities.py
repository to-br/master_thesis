import numpy as np
from numpy import random
import math
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import os
import sys
from scipy.signal import welch
 
"""old version - use utilities1.py"""

def calculate_noise(tau_si, C_m_si, Vm_SD, delta):
    """This function computes and returns the noise level given:
    tau_si : membrane time constant in SI-units
    C_m : membrane capacitance in SI-units
    Vm_SD : standard deviation of membrane potential
    delta : rate of change of current (how often new samples are drawn from Gaussian dist)"""
    
    SD_si = math.sqrt( (2 * C_m_si**2 * Vm_SD**2) / (delta * tau_si ))
    SD = SD_si / (1e-12)
    
    return SD

def transfer_V_to_I(V_target_si, f, tau_si=10e-3, R_si=100e6, E_m_si=-60e-3):
    """This function transfers a target membrane potential to the stimulating current 
    necessary to give this membrane potential during simulation
    V_target_si : Targeted membrane potential
    f : input frequency
    tau_si : membrane time constant in SI-units
    R_si : membrane resistance in SI-units
    E_m_si: resting potential in SI-units"""
    
    a_si = (-E_m_si+V_target_si) * math.sqrt(1 + (2*math.pi*f*tau_si)**2) / R_si
    a = a_si/(1e-12)
    
    return a


def save_figure(filename):
    """This function saves figures to the results-folder given a filename"""

    Path = os.getcwd()
    sys.path.append(Path)
    save_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(save_dir, exist_ok=True)
    file_index = 1
    while os.path.exists(os.path.join(save_dir, f"main_{file_index}.png")):
        file_index += 1
    
    save_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
 
    print(f"Saved plot as: {save_path}")
    

def psd(spike_times, sim_time, bin_size=0.25, segments=16):
    """This function computes the PSD-values given:
    spike_times : spike train, 
    sim_time : duration of simulation
    bin_size : frequency resolution
    segments : number of segments the signal is to be divided into
    It returns PSD-values of the spike train, and corresponding frequencies"""
    #creating bins with width = bin_size (ms)
    t_bins = np.arange(0, sim_time + bin_size, bin_size)
    
    # putting spike train into the bins 
    spike_counts, _ = np.histogram(spike_times, bins=t_bins)
    
    # Spike rate
    spike_rate = spike_counts / (bin_size / 1000.0)
    
    # Sampling rate (measurements per sec)
    fs = 1000.0 / bin_size 
    
    # using scipys Welch function to find frequencies and psd values. overlap fraction is 0.5 
    freqs, psd_values = welch(spike_rate, fs=fs, nperseg=len(spike_counts)//segments,
                              noverlap=(len(spike_counts)//segments)//2)
    
    # # Cut off high-frequency tail to avoid artifacts near Nyquist
    return freqs[:-250], psd_values[:-250]



def psd_Vm(Vm, sim_time, resolution, segments=16):
    """This function computes the PSD-values given:
    Vm : recorded membrane potential, 
    sim_time : duration of simulation
    bin_size : frequency resolution
    segments : number of segments the signal is to be divided into
    It returns PSD-values of the membrane potential, and corresponding frequencies"""
    
    fs = 1000.0 / resolution  # Sampling frequency [Hz]
    
    # using scipys Welch function to find frequencies and psd values. overlap fraction is 0.5 
    freqs, psd_values = welch(Vm, fs=fs, nperseg=len(Vm)//segments, noverlap=(len(Vm)//segments)//2)

    return freqs[:-250], psd_values[:-250]



def simulated_SNR(Vm, a, psd_values, f, freqs, arg_f, sim_time, sd, beat, spike_times, second_sine):
    """This function calculates the SNR. It also creates and returns dataframe to display SNR and other data,
    providing functionality and control throughout simulations
    Vm : Recorded membrane potential
    a: amplitude of input current
    psd_values: PSD of spike-train
    f : frequency of input current 
    freqs : list of frequencies corresponding to PSD-values
    arg_f : argument in freqs closest to input frequency
    sim_time : duration of simulation
    sd : noise level (standard deviation of current)
    beat : modulation frequency
    spike_times : list of times when action pot. is recorded
    second_sine : whether or not the second AC-generator is connected"""
    
    window = 5  # 5 Hz-window around f

    # Finding the indexes of frequencies around the input frequency
    noise_mask = (freqs >= freqs[arg_f] - window) & (freqs <= freqs[arg_f] + window) & (freqs != freqs[arg_f])

    
    max_fft_value = np.max(np.abs(psd_values))
    fft_value_f = np.abs(psd_values[arg_f])
    max_freq = freqs[np.argmax(np.abs(psd_values))]

    # Local noise average 
    noise_avg = np.mean(psd_values[noise_mask])
    
    # Computing SNR
    if noise_avg > 0:
        SNR = ((psd_values[arg_f]) / noise_avg)
    
    else:
        SNR = np.nan
    results_list = []
    
    # Creating dictionary to return results, separating between one and two AC-generators
    if second_sine:
        arg_beat = np.argmin(np.abs(freqs-beat))
        mask = (freqs >= freqs[arg_beat] - window) & (freqs <= freqs[arg_beat] + window) & (freqs != freqs[arg_beat])
        noise_avg_beat = np.mean((psd_values[mask]))
        SNR_beat =  (np.abs(psd_values[arg_beat]) / noise_avg_beat)
        
        results_list.append({
        "Input Frequency (Hz)": [f, f+beat],
        "Beat Frequency (Hz)": beat,
        "Current Amplitude (pA)": a,
        "Max Vm (mV)": np.max(Vm),
        "Max FFT Amplitude [(spikes per sec)/Hz]": np.round(max_fft_value,2),
        "PSD Amplitude for f [(spikes pr sec)/Hz]": round(fft_value_f,2),
        "PSD Amplitude for f2-f1 [(spikes pr sec)/Hz]": round(psd_values[arg_beat],2),
        "Frequency at max (Hz)": max_freq,
        "SNR (f)": round(SNR,2),
        "noise average": round(noise_avg,2),
        "Spike Rate (spikes/s)": (len(spike_times)/sim_time)*1000,
        "SNR (beat)": round(SNR_beat,1),
        "noise average (beat)" : round(noise_avg_beat,1),
        
    })
    else:
        results_list.append({
        "Input Frequency (Hz)": f,
        "Current Amplitude (pA)": a,
        "Max Vm (mV)": np.max(Vm),
        "Max FFT Amplitude [(spikes per sec)/Hz]": round(max_fft_value,2),
        "FFT Amplitude for f [(spikes pr sec)/Hz]": round(fft_value_f,2),
        "Frequency at max (Hz)": max_freq,
        "SNR": round(SNR,2),
        "noise average": round(noise_avg,2),
        "Spike Rate (spikes/s)": (len(spike_times)/sim_time)*1000
            
    })
        
    # returning dictionary as dataframe to display data
    return pd.DataFrame(results_list), noise_avg
