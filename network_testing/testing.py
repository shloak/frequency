from utils import *
import matplotlib.pyplot as plt
import numpy as np

def test_time_scaling():
    #all_ms = [[25, 25], [30, 30], [35, 35], [40, 40]]
    all_ms = [[a, a] for a in range(20, 60, 5)]
    bases, exps, dict_sizes, batch_size, SNRdB = [2, 3], [5, 5], [100, 3], 5, 8
    times1, times2 = [], []
    for ms in all_ms:
        trial1, trial2 = [], []
        for _ in range(5):
            (t1, t2), (all_preds, time_small), (all_preds_full, time_full) = frequency_detection(ms, bases, exps, dict_sizes, batch_size, SNRdB, num_iters=10)
            print(time_full, time_small)
            trial1.append(time_small)
            trial2.append(time_full)
        times1.append(np.median(trial1))
        times2.append(np.median(trial2))
    np.save('./data/times1', times1)
    np.save('./data/times2', times2)
    plt.plot([a[0] for a in all_ms], times1, '-bo')
    plt.plot([a[0] for a in all_ms], times2, '-ro')
    plt.show()
    
def test_freq_scaling():
    all_ms = [[a, a + 5] for a in range(35, 60, 5)] # 10
    bases, exps, dict_sizes, batch_size = [2, 3], [11, 7], [2500, 500], 10 
    N = (bases[0] ** exps[0]) * (bases[1] ** exps[1])
    snrs = [5, 4, 3, 2, 1, 0, -1, -2]
    NUM_TRIALS = 8
    for SNRdB in snrs:
        accs1, accs2 = [], []
        times1, times2 = [], []
        for ms in all_ms:
            indices = np.sort(np.random.choice(range(N), size=ms[0], replace=False))
            d1, d2 = generate_data_dicts(N, ms, bases, exps, dict_sizes[0], batch_size, SNRdB, indices) 
            t1, t2 = generate_data_dicts(N, ms, bases, exps, dict_sizes[1], batch_size, SNRdB, indices, boundaries=True)

            trial1, trial2 = [], []
            ti1, ti2 = [], []
            for _ in range(NUM_TRIALS):
                print(ms)
                (t1, t2), (all_preds, time_small), (all_preds_full, time_full) = frequency_detection(ms, bases, exps, dict_sizes, batch_size, SNRdB, 
                    num_iters=8000, layers=1, mle_indices=indices, train_dicts=(d1, d2), test_dicts=(t1, t2), verbose=True)
                ti1.append(time_small)
                ti2.append(time_full)
                trial1.append(calculate_accuracy(t1, all_preds, dict_sizes[1], batch_size))
                trial2.append(calculate_accuracy(t1, all_preds_full, dict_sizes[1], batch_size))

            accs1.append(np.max(trial1))
            accs2.append(np.max(trial2))
            times1.append(np.median(ti1))
            times2.append(np.median(ti2))
        #np.save('./data/accuracy_zerobit_SNR{}'.format(SNRdB), accs1)
        #np.save('./data/accuracy_twobit_SNR{}'.format(SNRdB), accs2)
        #np.save('./data/times_zerobit_SNR{}'.format(SNRdB), times1)
        #np.save('./data/times_twobit_SNR{}'.format(SNRdB), times2)
    #plt.plot([a[0] for a in all_ms], accs1, '-bo')
    #plt.plot([a[0] for a in all_ms], accs2, '-ro')
    #plt.show()

def test_mle_timing():
    all_ms = [[a, a + 5] for a in range(10, 60, 5)] 
    bases, exps, test_size = [2, 3], [11, 7], 10
    N = 100 #(bases[0] ** exps[0]) * (bases[1] ** exps[1])
    snrs = [-2, 5, 4, 3, 2, 1, 0, -1, -2]
    for SNRdB in snrs:
        sigma2 = get_sigma2_from_snrdb(SNRdB)
        for ms in all_ms:
            ms = [70, 75]
            sig_len = 10 #size_indices_lohi(N, ms, bases, exps)
            indices = np.sort(np.random.choice(range(N), size=sig_len, replace=False))
            test_signals = []
            test_freqs = []
            for i in range(test_size):
                index = np.random.randint(0, N) 
                test_freqs.append(index)
                w = 2 * np.pi * index / N
                test_signals.append([np.exp(1j*(w*ind)) + make_noise(sigma2, 1)[0] for ind in indices])
            acc, time = test_noisy_mle_random_inds(N, test_signals, test_freqs, indices, plotting=False)
            print(acc, time)

def test_kay_timing():
    all_ms = [[a, a + 5] for a in range(10, 2000, 250)] 
    bases, exps, test_size = [2, 3], [11, 7], 100
    N = (bases[0] ** exps[0]) * (bases[1] ** exps[1])
    snrs = [20, 5, 4, 3, 2, 1, 0, -1, -2]
    for SNRdB in snrs:
        sigma2 = get_sigma2_from_snrdb(SNRdB)
        for ms in all_ms:
            sig_len = size_indices_lohi(N, ms, bases, exps)
            print(sig_len)
            signals, freqs = make_batch_noisy(test_size, SNRdB, N, sig_len)
            acc, time = test_kays(signals, freqs, N, sig_len, T= 10*(exps[0] * exps[1])) # change to not one hot and use random indices for mle instead of sequential, but regardless acc is 0
            print(acc, time)

def test_mle_accuracy():
    all_ms = [[60, 65]] #[[a, a + 5] for a in range(10, 60, 5)] 
    bases, exps, test_size = [2, 3], [11, 7], 10
    N = (bases[0] ** exps[0]) * (bases[1] ** exps[1])
    snrs = [-2] #[-2, 5, 4, 3, 2, 1, 0, -1, -2]
    for SNRdB in snrs:
        sigma2 = get_sigma2_from_snrdb(SNRdB)
        for ms in all_ms:
            #ms = [70, 75]
            sig_len = size_indices_lohi(N, ms, bases, exps)
            indices = np.sort(np.random.choice(range(N), size=sig_len, replace=False))
            test_signals = []
            test_freqs = []
            for i in range(test_size):
                index = np.random.randint(0, N) 
                test_freqs.append(index)
                w = 2 * np.pi * index / N
                test_signals.append([np.exp(1j*(w*ind)) + make_noise(sigma2, 1)[0] for ind in indices])
            acc, time = test_noisy_mle_random_inds(N, test_signals, test_freqs, indices, plotting=False, verbose=True)
            print(acc, time)

test_mle_accuracy()
#test_mle_timing()
#test_freq_scaling()
#test_kay_timing()