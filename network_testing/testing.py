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
            acc, time = test_noisy_mle_random_inds_pool(N, test_signals, test_freqs, indices, plotting=False, verbose=True)
            print(acc, time)

def test_single_radix_boundaries():
    exps = [8, 10, 12, 14]
    base, m, dict_sizes, batch_size, SNRdB = 2, 25, [2500, 50000], 5, 4
    for exp in exps:
        (t1, t2), (all_preds, time_small), (all_preds_full, time_full) = frequency_detection_single_radix(m, base, exp, dict_sizes, batch_size, SNRdB, num_iters=8000, layers=1, mle_indices = [], train_dicts = [], test_dicts = [], verbose=True)
        calculate_accuracy(t1, all_preds, dict_sizes[1], batch_size)
        calculate_accuracy(t1, all_preds_full, dict_sizes[1], batch_size)
        wrong_actual, wrong_guess, bits_off, wrong_bits, freq_counts = get_miss_distribution(m, base, exp, t1, all_preds, dict_sizes[1], batch_size)
        wrong_actual_full, wrong_guess_full, bits_off_full, wrong_bits_full, freq_counts = get_miss_distribution(m, base, exp, t1, all_preds_full, dict_sizes[1], batch_size)

        np.save('./data/boundary_testing_big/wrong_actual_small_exp{}'.format(exp), wrong_actual)
        np.save('./data/boundary_testing_big/wrong_guess_small_exp{}'.format(exp), wrong_guess)
        np.save('./data/boundary_testing_big/bits_off_small_exp{}'.format(exp), bits_off)
        np.save('./data/boundary_testing_big/wrong_bits_small_exp{}'.format(exp), wrong_bits)
        np.save('./data/boundary_testing_big/wrong_actual_full_exp{}'.format(exp), wrong_actual_full)
        np.save('./data/boundary_testing_big/wrong_guess_full_exp{}'.format(exp), wrong_guess_full)
        np.save('./data/boundary_testing_big/bits_off_full_exp{}'.format(exp), bits_off_full)
        np.save('./data/boundary_testing_big/wrong_bits_full_exp{}'.format(exp), wrong_bits_full)
        np.save('./data/boundary_testing_big/freq_counts_exp{}'.format(exp), freq_counts)

def test_shift_method():
    exps = [8, 10, 12, 14]
    #exp = 8
    base, m, dict_sizes, batch_size, SNRdB = 2, 25, [2500, 50000], 5, 4
    #shifts = [1/8, 1/16] # how much of a circle to shift clockwise (eg input freq is f0 - 2pi*shift)
    shifts = [2 ** -3, 2 ** -5, 2 ** -7, 2 ** -9] 
    for exp, shift in zip(exps, shifts):
    #for shift in shifts:
        N = base ** exp
        (t1, t2), (all_preds, time_small), (all_preds_full, time_full), shift_all_preds_full, indices = \
             frequency_detection_single_radix(m, base, exp, dict_sizes, batch_size, SNRdB, num_iters=8000, shift=shift, layers=1, mle_indices = [], train_dicts = [], test_dicts = [], verbose=False)
        final_preds = compare_shifted(t1, all_preds_full, shift_all_preds_full, dict_sizes[1], batch_size, indices, N, [base, 2], [exp, 0], shift=shift) # shifted predictions
        print('SHIFT:', shift)
        calculate_accuracy(t1, all_preds_full, dict_sizes[1], batch_size)
        calculate_accuracy(t1, final_preds, dict_sizes[1], batch_size)

        wrong_actual_full, wrong_guess_full, bits_off_full, wrong_bits_full, freq_counts = get_miss_distribution(m, base, exp, t1, all_preds_full, dict_sizes[1], batch_size)
        wrong_actual_shift, wrong_guess_shift, bits_off_shift, wrong_bits_shift, freq_counts = get_miss_distribution(m, base, exp, t1, final_preds, dict_sizes[1], batch_size)
        
        np.save('./data/boundary_testing_shifted_t3/wrong_actual_shift_exp{}'.format(exp), wrong_actual_shift)
        np.save('./data/boundary_testing_shifted_t3/wrong_guess_shift_exp{}'.format(exp), wrong_guess_shift)
        np.save('./data/boundary_testing_shifted_t3/bits_off_shift_exp{}'.format(exp), bits_off_shift)
        np.save('./data/boundary_testing_shifted_t3/wrong_bits_shift_exp{}'.format(exp), wrong_bits_shift)
        np.save('./data/boundary_testing_shifted_t3/wrong_actual_full_exp{}'.format(exp), wrong_actual_full)
        np.save('./data/boundary_testing_shifted_t3/wrong_guess_full_exp{}'.format(exp), wrong_guess_full)
        np.save('./data/boundary_testing_shifted_t3/bits_off_full_exp{}'.format(exp), bits_off_full)
        np.save('./data/boundary_testing_shifted_t3/wrong_bits_full_exp{}'.format(exp), wrong_bits_full)
        np.save('./data/boundary_testing_shifted_t3/freq_counts_exp{}'.format(exp), freq_counts)
        


def test_shift_method_accuracy():
    exp = 8
    base, dict_sizes, batch_size, SNRdB = 2, [2500, 5000], 5, 2
    ms = [i for i in range(8, 22, 2)]
    shift = 1/8
    NUM_TRIALS = 4
    all_accs = []
    N = base ** exp
    for m in ms:

        trial_accs = []
        for _ in range(NUM_TRIALS):

            (t1, t2), (all_preds, time_small), (all_preds_full, time_full), shift_all_preds_full, indices = \
                frequency_detection_single_radix(m, base, exp, dict_sizes, batch_size, SNRdB, num_iters=8000, shift=shift, layers=1, mle_indices = [], train_dicts = [], test_dicts = [], verbose=False)
            final_preds = compare_shifted(t1, all_preds_full, shift_all_preds_full, dict_sizes[1], batch_size, indices, N, [base, 2], [exp, 0], shift=shift) # shifted predictions

            trial_accs.append(calculate_accuracy(t1, final_preds, dict_sizes[1], batch_size))
    
        all_accs.append(np.max(trial_accs))


    np.save('./data/accuracy_shifted/exp{}snr{}accs_shifted'.format(exp, SNRdB), all_accs)    





if __name__ == '__main__':    
    test_shift_method_accuracy()
    #test_single_radix_boundaries()
    #test_mle_accuracy()
    #test_mle_timing()
    #test_freq_scaling()
    #test_kay_timing()

