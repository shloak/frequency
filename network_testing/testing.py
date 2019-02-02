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
    all_ms = [[a, a + 5] for a in range(10, 100, 5)] 
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
            t1, t2 = generate_data_dicts(N, ms, bases, exps, dict_sizes[1], batch_size, SNRdB, indices)

            trial1, trial2 = [], []
            ti1, ti2 = [], []
            for _ in range(NUM_TRIALS):
                print(ms)
                (t1, t2), (all_preds, time_small), (all_preds_full, time_full) = frequency_detection(ms, bases, exps, dict_sizes, batch_size, SNRdB, 
                    num_iters=8000, layers=1, mle_indices=indices, train_dicts=(d1, d2), test_dicts=(t1, t2))
                ti1.append(time_small)
                ti2.append(time_full)
                trial1.append(calculate_accuracy(t1, all_preds, dict_sizes[1], batch_size))
                trial2.append(calculate_accuracy(t1, all_preds_full, dict_sizes[1], batch_size))
            accs1.append(np.max(trial1))
            accs2.append(np.max(trial2))
            times1.append(np.median(ti1))
            times2.append(np.median(ti2))
        np.save('./data/accuracy_small_SNR{}'.format(SNRdB), accs1)
        np.save('./data/accuracy_full_SNR{}'.format(SNRdB), accs2)
        np.save('./data/times_small_SNR{}'.format(SNRdB), times1)
        np.save('./data/times_full_SNR{}'.format(SNRdB), times2)
    #plt.plot([a[0] for a in all_ms], accs1, '-bo')
    #plt.plot([a[0] for a in all_ms], accs2, '-ro')
    #plt.show()
test_freq_scaling()