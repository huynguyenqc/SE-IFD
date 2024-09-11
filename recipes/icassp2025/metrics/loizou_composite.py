import numpy as np
import pesq
import scipy.fftpack as fft
from typing import Tuple
from scipy.linalg import toeplitz


def composite(clean_t: np.ndarray, enhanced_t: np.ndarray, fs: float) -> Tuple[float, float, float, float]:
    alpha = 0.95
    
    # Match the lengths of the two files
    sig_len = min(len(clean_t), len(enhanced_t))
    clean_t = clean_t[:sig_len] + np.finfo(float).eps
    enhanced_t = enhanced_t[:sig_len] + np.finfo(float).eps
    
    # Compute WSS measure
    wss_dist_vec = wss(clean_t, enhanced_t, fs)
    wss_dist_vec = np.sort(wss_dist_vec)
    wss_dist = np.mean(wss_dist_vec[:int(len(wss_dist_vec) * alpha)])
    
    # Compute LLR measure
    LLR_dist = llr(clean_t, enhanced_t, fs)
    LLRs = np.sort(LLR_dist)
    llr_mean = np.mean(LLRs[:int(len(LLRs) * alpha)])
    
    # Compute SNRseg
    snr_dist, segsnr_dist = snr(clean_t, enhanced_t, fs)
    segSNR = np.mean(segsnr_dist)
    
    # Placeholder for PESQ score
    pesq_mos = pesq.pesq(mode='wb', fs=fs, ref=clean_t, deg=enhanced_t)
    
    # Compute composite measures
    Csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_mos - 0.009 * wss_dist
    Cbak = 1.634 + 0.478 * pesq_mos - 0.007 * wss_dist + 0.063 * segSNR
    Covl = 1.594 + 0.805 * pesq_mos - 0.512 * llr_mean - 0.007 * wss_dist
    
    return Csig, Cbak, Covl, segSNR, (llr_mean, pesq_mos, wss_dist)


def wss(clean_t: np.ndarray, enhanced_t: np.ndarray, fs: float) -> float:
    clean_length = len(clean_t)
    
    assert clean_length == len(enhanced_t), 'Clean and enhanced signals must have the same length!'
    
    winlength = int(round(30 * fs / 1000))
    skiprate = winlength // 4
    num_crit = 25
    n_fft = 2 ** int(np.ceil(np.log2(2 * winlength)))
    n_fftby2 = n_fft // 2
    
    cent_freq = [50.0, 120.0, 190.0, 260.0, 330.0, 400.0, 470.0, 540.0,
                 617.372, 703.378, 798.717, 904.128, 1020.38, 1148.30,
                 1288.72, 1442.54, 1610.70, 1794.16, 1993.93, 2211.08,
                 2446.71, 2701.97, 2978.04, 3276.17, 3597.63]
    
    bandwidth = [70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 77.3724, 86.0056, 
                 95.3398, 105.411, 116.256, 127.914, 140.423, 153.823, 168.154, 
                 183.457, 199.776, 217.153, 235.631, 255.255, 276.072, 298.126, 
                 321.465, 346.136]
    
    bw_min = bandwidth[0]
    max_freq = fs / 2
    min_factor = np.exp(-30.0 / (2.0 * 2.303))
    
    crit_filter = np.zeros((num_crit, n_fftby2))
    
    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * n_fftby2
        bw = (bandwidth[i] / max_freq) * n_fftby2
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        j = np.arange(n_fftby2)
        crit_filter[i, :] = np.exp(-11 * (((j - np.floor(f0)) / bw) ** 2) + norm_factor)
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > min_factor)
    
    num_frames = clean_length // skiprate - (winlength // skiprate)
    start = 0
    window = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(winlength) + 1) / (winlength + 1)))
    distortion = np.zeros(num_frames)
    
    for frame_count in range(num_frames):
        clean_frame = clean_t[start:start + winlength] * window
        processed_frame = enhanced_t[start:start + winlength] * window
        
        clean_spec = np.abs(fft.fft(clean_frame, n_fft)) ** 2
        processed_spec = np.abs(fft.fft(processed_frame, n_fft)) ** 2
        
        clean_energy = np.zeros(num_crit)
        processed_energy = np.zeros(num_crit)
        
        for i in range(num_crit):
            clean_energy[i] = np.sum(clean_spec[:n_fftby2] * crit_filter[i, :])
            processed_energy[i] = np.sum(processed_spec[:n_fftby2] * crit_filter[i, :])
        
        clean_energy = 10 * np.log10(np.maximum(clean_energy, 1E-10))
        processed_energy = 10 * np.log10(np.maximum(processed_energy, 1E-10))
        
        clean_slope = clean_energy[1:num_crit] - clean_energy[:num_crit-1]
        processed_slope = processed_energy[1:num_crit] - processed_energy[:num_crit-1]
        
        clean_loc_peak = np.zeros(num_crit - 1)
        processed_loc_peak = np.zeros(num_crit - 1)
        
        for i in range(num_crit - 1):
            if clean_slope[i] > 0:
                n = i
                while n < num_crit-1 and clean_slope[n] > 0:
                    n += 1
                clean_loc_peak[i] = clean_energy[n - 1]
            else:
                n = i
                while n >= 0 and clean_slope[n] <= 0:
                    n -= 1
                clean_loc_peak[i] = clean_energy[n + 1]
            
            if processed_slope[i] > 0:
                n = i
                while n < num_crit-1 and processed_slope[n] > 0:
                    n += 1
                processed_loc_peak[i] = processed_energy[n - 1]
            else:
                n = i
                while n >= 0 and processed_slope[n] <= 0:
                    n -= 1
                processed_loc_peak[i] = processed_energy[n + 1]
        
        dBMax_clean = np.max(clean_energy)
        dBMax_processed = np.max(processed_energy)
        
        Kmax = 20
        Klocmax = 1
        
        Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[:num_crit - 1])
        Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peak - clean_energy[:num_crit - 1])
        W_clean = Wmax_clean * Wlocmax_clean
        
        Wmax_processed = Kmax / (Kmax + dBMax_processed - processed_energy[:num_crit - 1])
        Wlocmax_processed = Klocmax / (Klocmax + processed_loc_peak - processed_energy[:num_crit - 1])
        W_processed = Wmax_processed * Wlocmax_processed
        
        W = (W_clean + W_processed) / 2.0
        
        distortion[frame_count] = np.sum(W * (clean_slope - processed_slope) ** 2)
        distortion[frame_count] /= np.sum(W)
        
        start += skiprate
    
    return distortion


def llr(clean_t: np.ndarray, enhanced_t: np.ndarray, fs: float) -> float:
    clean_length = len(clean_t)
    
    assert clean_length == len(enhanced_t), 'Both Speech Files must be same length.'
    
    winlength = int(round(30 * fs / 1000))
    skiprate = winlength // 4
    P = 10 if fs < 10000 else 16
    
    num_frames = clean_length // skiprate - (winlength // skiprate)
    start = 0
    window = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(winlength) + 1) / (winlength + 1)))
    distortion = np.zeros(num_frames)
    
    for frame_count in range(num_frames):
        clean_frame = clean_t[start:start + winlength] * window
        processed_frame = enhanced_t[start:start + winlength] * window

        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)
        
        # R_clean = np.correlate(clean_frame, clean_frame, mode='full')[winlength-1:]
        # R_clean = R_clean[:P+1]
        # R_clean = toeplitz(R_clean[:-1])
        # r_clean = R_clean[:, -1]
        
        # a_clean = np.linalg.pinv(R_clean).dot(r_clean)
        # a_clean = np.concatenate(([1], -a_clean))
        
        # R_processed = np.correlate(processed_frame, processed_frame, mode='full')[winlength-1:]
        # R_processed = R_processed[:P+1]
        # R_processed = toeplitz(R_processed[:-1])
        # r_processed = R_processed[:, -1]
        
        # a_processed = np.linalg.pinv(R_processed).dot(r_processed)
        # a_processed = np.concatenate(([1], -a_processed))
        
        numerator   = A_processed.dot(toeplitz(R_clean)).dot(A_processed.T)
        denominator = A_clean.dot(toeplitz(R_clean)).dot(A_clean.T)
        # term1 = np.sum(a_processed.dot(R_clean) * a_processed)
        # term2 = np.sum(a_clean.dot(R_clean) * a_clean)
        
        distortion[frame_count] = np.log(numerator / denominator)
        
        start += skiprate
    
    return distortion


def lpcoeff(speech_frame, model_order):
    # ----------------------------------------------------------
    # (1) Compute Autocorrelation Lags
    # ----------------------------------------------------------
    
    winlength = len(speech_frame)
    R = np.zeros(model_order + 1)
    for k in range(model_order + 1):
        R[k] = np.sum(speech_frame[:winlength - k] * speech_frame[k:])
    
    # ----------------------------------------------------------
    # (2) Levinson-Durbin
    # ----------------------------------------------------------
    
    a = np.ones(model_order)
    E = np.zeros(model_order + 1)
    E[0] = R[0]
    rcoeff = np.zeros(model_order)
    
    for i in range(model_order):
        a_past = a[:i]
        sum_term = np.sum(a_past * R[i:0:-1])
        rcoeff[i] = (R[i + 1] - sum_term) / E[i]
        a[i] = rcoeff[i]
        if i > 0:
            a[:i] = a_past - rcoeff[i] * a_past[::-1]
        E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]
    
    acorr = R
    refcoeff = rcoeff
    lpparams = np.concatenate(([1], -a))
    
    return acorr, refcoeff, lpparams


def snr(clean_t: np.ndarray, enhanced_t: np.ndarray, fs: float) -> float:
    clean_length = len(clean_t)
    processed_length = len(enhanced_t)
    
    if clean_length != processed_length:
        raise ValueError('Both Speech Files must be same length.')
    
    winlength = int(round(30 * fs / 1000))
    skiprate = winlength // 4
    
    num_frames = clean_length // skiprate - (winlength // skiprate)
    start = 0
    window = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(winlength) + 1) / (winlength + 1)))
    
    segsnr = np.zeros(num_frames)
    
    for frame_count in range(num_frames):
        clean_frame = clean_t[start:start + winlength] * window
        processed_frame = enhanced_t[start:start + winlength] * window
        
        signal_energy = np.sum(clean_frame ** 2)
        noise_energy = np.sum((clean_frame - processed_frame) ** 2)
        
        segsnr[frame_count] = 10 * np.log10(signal_energy / (noise_energy + np.finfo(float).eps) + np.finfo(float).eps)
        
        start += skiprate
    
    snr = np.mean(segsnr)
    
    return snr, segsnr
