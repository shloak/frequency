import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

def make_signal(w,theta,n):
    """
    Assumes normalized amplitude
    """
    t = np.arange(n)
    signal = np.exp(1j*(w*t + theta))
    return signal

def make_signal_random(w,theta,N,m):
    sig = make_signal(w, theta, N)
    chosen_indices = np.sort(np.random.choice(range(N), size=m, replace=False))
    return (np.take(sig, chosen_indices), chosen_indices)

def make_signal_2power(w,theta,N,m):
    sig = make_signal(w, theta, N)
    chosen_indices = list(set([(j * 2**i) % N for j in range(m) for i in range(int(np.log2(N)))]))
    #print('length: ', len(chosen_indices))
    return (np.take(sig, chosen_indices), chosen_indices)

def make_noise(sigma2,n):
    noise_scaling = np.sqrt(sigma2/2)
    # noise is complex valued
    noise  = noise_scaling*np.random.randn(n) + 1j*noise_scaling*np.random.randn(n)
    return noise
    

def make_noisy_signal(w,theta,SNRdb,n):
    sigma2 = get_sigma2_from_snrdb(SNRdb)
    signal = make_signal(w,theta,n)
    noise  = make_noise(sigma2,n)
    return signal + noise

def make_noisy_signal_random(w, theta, SNRdb, N, m):
    sigma2 = get_sigma2_from_snrdb(SNRdb)
    signal, inds = make_signal_random(w, theta, N, m)
    noise = make_noise(sigma2, m)
    return signal+noise, inds

def make_noisy_signal_2power(w, theta, SNRdb, N, m):
    sigma2 = get_sigma2_from_snrdb(SNRdb)
    signal, inds = make_signal_2power(w, theta, N, m)
    noise = make_noise(sigma2, len(inds))
    return signal+noise, inds

def make_batch_noisy_random(batch_size, SNRdb, N, m):
    signals, freqs, inds = [], [], []
    for i in range(batch_size):
        freq = np.random.randint(0, N)
        w = (2 * np.pi * freq / N) % (2 * np.pi)
        sig, ind = make_noisy_signal_random(w, 0, SNRdb, N, m)
        signals.append(sig)
        freqs.append(freq)
        inds.append(ind)
    return signals, one_hot(N, batch_size, freqs), inds
    
def make_batch_noisy_2power(batch_size, SNRdb, N, m):
    signals, freqs, inds = [], [], []
    for i in range(batch_size):
        freq = np.random.randint(0, N)
        w = (2 * np.pi * freq / N) % (2 * np.pi)
        sig, ind = make_noisy_signal_2power(w, 0, SNRdb, N, m)
        signals.append(sig)
        freqs.append(freq)
        inds.append(ind)
    return signals, one_hot(N, batch_size, freqs), inds

    
# N = divisor of w0
# m = num samples
def make_batch_noisy(batch_size, SNRdb, N, m, binary=False):
    signals, freqs = [], []
    for i in range(batch_size):
        freq = np.random.randint(0, N)
        w = (2 * np.pi * freq / N) % (2 * np.pi)
        sig = make_noisy_signal(w, 0, SNRdb, m)
        signals.append(sig)
        freqs.append(freq)
    if binary:
        return signals, make_binary(freqs, N), one_hot(N, batch_size, freqs)
    return signals, one_hot(N, batch_size, freqs)

# N = divisor of w0
# m = num samples
def make_batch_noisy_lohi(batch_size, SNRdb, N, m):
    freqs = []
    freqs.append(np.random.randint(0, N))
    test_signals, test_freqs = make_noisy_lohi(SNRdB, N, m, freqs[-1])
    for i in range(1, batch_size):
        freqs.append(np.random.randint(0, N))
        a, b = make_noisy_lohi(SNRdB, N, m, freqs[-1])
        test_signals.extend(a)
        test_freqs.extend(b)
    return test_signals, test_freqs, freqs

def make_noisy_lohi(SNRdb, N, m, freq):
    signals, vals = [], []
    steps = int(np.log2(N))
    w = (2 * np.pi * freq / N) % (2 * np.pi)
    sig = make_noisy_signal(w, 0, SNRdb, m * (2**steps))
    for i in range(int(np.log2(N))):
        signals.append([sig[a * (2**i)] for a in range(m)])
        if (freq * (2**i)) % (N) < N / 2:
            vals.append([1, 0])
        else:
            vals.append([0, 1])
    return signals, vals
        

def make_batch_singleton(batch_size, SNRdb, N, m, default=-1): # 0 = zero, 1 = single, 2 = multi
    signals, freqs = [], []
    sigma2 = get_sigma2_from_snrdb(SNRdB)
    for i in range(batch_size):
        val = np.random.poisson(0.79)
        if default >= 0:
            val = default
        if val == 0:
            signals.append(make_noise(0, m))
            freqs.append([1, 0, 0])
        if val == 1:
            signals.append(make_noisy_signal(2 * np.pi * np.random.randint(0, N) / N, 0, SNRdB, m))
            freqs.append([0, 1, 0])
        if val >= 2:
            signal = make_signal(2 * np.pi * np.random.randint(0, N) / N, 0, m)
            for i in range(val - 1):
                signal += make_signal(2 * np.pi * np.random.randint(0, N) / N, 0, m)
            signals.append(signal + make_noise(sigma2, m))
            freqs.append([0, 0, 1])
    return signals, freqs

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

def kays_singleton_accuracy(test_signals, test_freqs, N):
    diffs = [s - make_signal(kays_method(s), 0, N) for s in test_signals]
    thresh, single_acc, other_acc, best_thresh = 0.0, 0, 0, 0
    best = 0
    for i in range(150):
        vals = [(sum(np.absolute(s)) / N) < thresh for s in diffs]
        corr = [1 for i in range(len(test_freqs)) if (test_freqs[i] == [0, 1, 0] and vals[i] == 1) or ((test_freqs[i] != [0, 1, 0] and vals[i] == 0))]
        corr = sum(corr)
        #single = sum([vals[d] for d in range(len(vals)) if test_freqs[d] == [0, 1, 0]]) / len([vals[d] for d in range(len(vals)) if test_freqs[d] == [0, 1, 0]])
        #other = sum([not vals[d] for d in range(len(vals)) if test_freqs[d] != [0, 1, 0]]) / len([vals[d] for d in range(len(vals)) if test_freqs[d] != [0, 1, 0]])        
        #if single*2 + other > single_acc*2 + other_acc and single > 0.2 and other > 0.2:
        #    single_acc = single
        #    other_acc = other
        #    best_thresh = thresh
        if corr > best:
            best = corr
            best_thresh = thresh
        thresh += 0.05
    print('thresh: ', best_thresh)
    return best / len(test_signals)

def test_min_thresh(signal, N, m, index, T):
    min_val, min_ind = np.inf, index
    for i in range(index - T//2, index + T//2 + 1):
        sig = make_signal((i * 2*np.pi/N), 0, m)
        resid = np.vdot(sig - signal, sig - signal)
        if resid < min_val:
            min_val = resid
            min_ind = i
    #print('thresh: ', min_ind)
    return min_ind
        

def test_kays(signals, freqs, N, m, T=5):
    count, count2 = 0, 0
    for sig, freq in zip(signals, freqs):
        res = kays_method(sig)
        res = int(round(res * N / (2 * np.pi))) % N
        #print('kay: ', res)
        if np.argmax(freq) == res:
            count += 1
        if np.argmax(freq) == int(test_min_thresh(sig, N, m, res, T)):
            count2 += 1
        #print('actual: ', np.argmax(freq))
    return count2 / len(signals)

def test_mle(signals, freqs, N, m):
    count = 0
    for sig, freq in zip(signals, freqs):
        cleans = [make_signal(np.pi * 2 * w / N, 0, m) for w in range(N)]
        dots = [np.absolute(np.vdot(sig, clean)) for clean in cleans]
        if np.argmax(dots) == np.argmax(freq):
            count += 1
    return count / len(signals)
    
def make_binary(freqs, N):
    w = math.ceil(np.log2(N))
    return [[int(a) for a in list(np.binary_repr(f, width=w))] for f in freqs] 

def binary_to_int(binary_string):
    return tf.reduce_sum(
    tf.cast(tf.reverse(tensor=binary_string, axis=[0]), dtype=tf.int64)
    * 2 ** tf.range(tf.cast(tf.size(binary_string), dtype=tf.int64)))
    '''y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y'''

def hamming(pred, act):
    return np.count_nonzero(pred != act)

def one_hot(N, batch_size, freqs):
    freqs_one_hot = np.zeros((batch_size, N))
    freqs_one_hot[np.arange(batch_size), freqs] = 1
    return freqs_one_hot

def test_noisy_mle(N, m, signals, freqs):
    count = 0  
    cleans = [make_signal(2*np.pi*i/N, 0, m) for i in range(N)]
                     
    for index in range(len(signals)):
        dots = [np.absolute(np.vdot(signals[index], cleans[i])) for i in range(N)]
        if np.argmax(freqs[index]) == np.argmax(dots):
            count += 1
    return count / len(freqs)

def test_noisy_mle_random(N, m, signals, freqs, inds):
    count = 0  
                     
    for index in range(len(signals)):
        cleans = [np.take(make_signal(2*np.pi*i/N, 0, N), inds[index]) for i in range(N)]
        dots = [np.absolute(np.vdot(signals[index], cleans[i])) for i in range(N)]
        if np.argmax(freqs[index]) == np.argmax(dots):
            count += 1
    return count / len(freqs)

def bit_to_freq(bits, N):
    possible = [i for i in range(N)]
    for b in bits:
        if b[0]:
            possible = possible[:len(possible)//2]
        else:
            possible = possible[len(possible)//2:]
    return possible[0]

# indices are constant
def make_noisy_signal_random_inds(w, theta, SNRdb, N, m, inds):
    sigma2 = get_sigma2_from_snrdb(SNRdb)
    signal = make_signal(w, theta, N)
    noise = make_noise(sigma2, m)
    return np.take(signal, inds) + noise

def make_batch_noisy_random_inds(batch_size, SNRdb, N, m, inds):
    signals, freqs = [], []
    for i in range(batch_size):
        freq = np.random.randint(0, N)
        w = (2 * np.pi * freq / N) % (2 * np.pi)
        sig = make_noisy_signal_random_inds(w, 0, SNRdb, N, m, inds)
        signals.append(sig)
        freqs.append(freq)
    return signals, one_hot(N, batch_size, freqs)

# indices are constant
def test_noisy_mle_random_inds(N, m, signals, freqs, inds, cleans):
    count = 0  
                     
    for index in range(len(signals)):
        dots = [np.absolute(np.vdot(signals[index], cleans[i])) for i in range(N)]
        if np.argmax(freqs[index]) == np.argmax(dots):
            count += 1
    return count / len(freqs)

# test mle detection for singletons - same random m samples

#snrs = [4, 2, 0, -2] #[10, 8, 6, 4, 2, 0, -2]
snrs = [-2, -6, -10, -14, -18]
N = 16384 
m = 166
batch_size = 1500
#SNRdB = 2
indices = np.sort(np.random.choice(range(N), size=m, replace=False))
cleans = [np.take(make_signal(2*np.pi*i/N, 0, N), indices) for i in range(N)]

res = []
#Ms = [100, 80, 60, 40, 20]

for SNRdB in snrs:
    print(SNRdB)
    #indices = np.sort(np.random.choice(range(N), size=m, replace=False))
    #cleans = [np.take(make_signal(2*np.pi*i/N, 0, N), indices) for i in range(N)]
    test_signals, test_freqs = make_batch_noisy_random_inds(batch_size, SNRdB, N, m, indices)
    res.append(test_noisy_mle_random_inds(N, m, test_signals, test_freqs, indices, cleans))
    print(res[-1])
    
plt.plot(snrs, res)
plt.title('MLE: 16384 possible frequencies, 166 samples')
plt.show()
#np.save('./data/divide_conquer/snrs_mle_16384_166', snrs)
#np.save('./data/divide_conquer/mle_random_16384_166', res)
    

 