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
    
test_time_scaling()
