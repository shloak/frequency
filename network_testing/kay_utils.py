import numpy as np
import math
from functools import reduce
import time
from multiprocessing import Pool

def make_signal(w,theta,n,skip=1,shift=0):
    """
    Assumes normalized amplitude
    """
    t = np.arange(0,n*skip,skip)+shift
    signal = np.exp(1j*(w*t + theta))
    return signal

def make_noise(sigma2,n):
    noise_scaling = np.sqrt(sigma2/2)
    # noise is complex valued
    noise  = noise_scaling*np.random.randn(n) + 1j*noise_scaling*np.random.randn(n)
    return noise

def make_noisy_signal(w,theta,sigma2,n,skip=1):
    signal = make_signal(w,theta,n,skip)
    noise  = make_noise(sigma2,n)
    return signal + noise

def get_sigma2_from_snrdb(SNR_db):
    return 10**(-SNR_db/10)

def kay_weights(N):
    scaling = (3.0/2)*N/(N**2 - 1)
    w = [1 - ((i - (N/2 - 1))/(N/2))**2 for i in range(N-1)]
    return scaling*np.array(w)

def kays_method(my_signal):
    N = len(my_signal)
    w = kay_weights(N)
    
    angle_diff = np.angle(np.conj(my_signal[0:-1])*my_signal[1:])
    need_to_shift = np.any(angle_diff < -np.pi/2)
    if need_to_shift:    
        neg_idx = angle_diff < 0
        angle_diff[neg_idx] += np.pi*2
    
    return w.dot(angle_diff)

def successive_estimation(signal_chain,N):
    location_bis = 0
    num_chains = len(signal_chain)
    # from radians to location
    factor = N/2/np.pi
    # how many frequencies wrap to the location
    nwrap = 1
    
    for chain in signal_chain:
        temp_location = (kays_method(chain)*factor) %  N
        
        loc_update = temp_location/nwrap - location_bis
        
        r = loc_update - round((loc_update * nwrap)/N)*N/nwrap
        
        location_bis += r
        nwrap *= 2
        
    return location_bis % N

def omega_to_location(w,N):
    return w*N/2/np.pi

def make_kay_chains(w0, num_chains, num_samples_per_chain,snrdb):
    # this is a list of arrays
    # it has num_chains many elements
    # each element is an array of length num_samples_per_chain
    chains = []
    # noise variance
    sigma2 = get_sigma2_from_snrdb(snrdb)
    
    for i in range(num_chains):
        # first chain gets samples with skipping 1
        # second chain gets samples with skipping 2, etc...
        skip = 2**i
        # skip needs to be passed
        signal = make_signal(w0,0,num_samples_per_chain,skip)
        noise  = make_noise(sigma2,num_samples_per_chain)
        noisy_signal = signal + noise
        # append to the chain
        chains.append(noisy_signal)
    return chains

def make_our_chains(w0, n, num_samples_per_chain, snrdb):
    '''
    n = what power of two is the signal
    '''
    # chains is a list of lists
    # it has n elements
    # each element is a list of tuples (sample1, sample2)
    chains = []
    # sampling_points is a list of lists
    # it has n elements
    # each element is a list of tuples (sampling_location_1, sampling_location_2)
    sampling_points = []
    sigma2 = get_sigma2_from_snrdb(snrdb)
    
    # you start by skipping signal_length/2
    skip = 2**(n-1)
    
    for i in range(n):
        current_chain = []
        current_sampling_points = []
        
        # poisition of the first sample
        shift = 0
        for i in range(num_samples_per_chain):
            signal = make_signal(w0,0,2,skip,shift)
            noise  = make_noise(sigma2,2)
            noisy_signal = signal + noise
            current_chain.append(tuple(noisy_signal))
            current_sampling_points.append(tuple([shift,shift+skip]))
            shift += 1
        chains.append(current_chain)
        sampling_points.append(current_sampling_points)
        # halve the skip
        skip /= 2
    return chains, sampling_points

def estimate_bit(chain, sampling_points, ref_w0):
    '''
    chain: pairs of samples obtained from the chain
    sampling_points: the points at which the samples are obtained
    ref_w0: the reference frequency obtained so far
    '''
    # complex valued estimate of the sign
    d = 0
    # number of pairs in a chain
    len_chain = len(chain)
    # go over the chain
    for i in range(len_chain):
        # where the samples are taken
        s1,s2 = sampling_points[i]
        # the sample values
        a,b = chain[i]
        # rotate the frequency based on the reference
        a = a*np.exp(-1j*ref_w0*s1)
        b = b*np.exp(-1j*ref_w0*s2)
        # check the sign
        d += b*np.conj(a)
    # average the sign
    # (this would change if there is a covariance between estimates)
    # (we get around this by taking non-overlapping samples)
    d /= len_chain
    # sign (returns -1 or 1)
    return np.sign(np.real(d))

def our_method(all_chains, all_sampling_points):
    # start with reference 0
    ref_w0 = 0
    n = len(all_chains)
    # signal length
    N = 2**n
    # go over chains
    for i in range(n):
        current_chain = all_chains[i]
        current_smapling_points = all_sampling_points[i]
        # estimate the bit
        current_bit = estimate_bit(current_chain,
                                   current_smapling_points,
                                   ref_w0)
        # if the current bit is -1 it means we are odd with
        # respect to ith bit of the frequency
        if current_bit == -1:
            ref_w0 += (2**i)*(2*np.pi)/N
    return ref_w0

def test_kays_method(N, n, SNRdB, total_samples, test_size):
    correct = 0
    for i in range(test_size):
        f0 = np.random.randint(0, N)
        w0 = (2 * np.pi * f0 / N)
        a = omega_to_location(kays_method(make_kay_chains(w0, 1, total_samples, SNRdB)[0]), N)
        
        b = round(a) % N
        correct += (b == f0)
        
    print(correct / test_size)   
    return correct / test_size

def test_successive_estimation(N, n, SNRdB, num_per_chain, test_size):
    correct = 0
    total_samples = n * num_per_chain
    for i in range(test_size):
        f0 = np.random.randint(0, N)
        w0 = (2 * np.pi * f0 / N)

        A = make_kay_chains(w0, n, num_per_chain, SNRdB)
        a = successive_estimation(A,N)
        
        b = round(a) % N
        correct += (b == f0)

    print(correct / test_size)    
    return correct / test_size

def test_our_method(N, n, SNRdB, num_per_chain, test_size):
    correct = 0
    total_samples = n * num_per_chain

    for i in range(test_size):
        f0 = np.random.randint(0, N)
        w0 = (2 * np.pi * f0 / N)

        A, B = make_our_chains(w0, n, num_per_chain, SNRdB)
        w0_est = our_method(A, B)

        a = omega_to_location(w0_est, N)    
        b = round(a)
        correct += (b == f0)        
        
    print(correct / test_size)   
    return correct / test_size
