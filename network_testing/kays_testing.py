from kay_utils import *
import matplotlib.pyplot as plt
import numpy as np

def method_accuracy_comparison():
    exps = [10, 12, 14]
    SNRdB = 2

    all_kay_accs = []
    all_successive_accs = []
    all_our_accs = []

    for n in exps:
        test_size = 15 * (2 ** n)
        kay_accs = []
        successive_accs = []
        our_accs = []

        nums = [n // 2, n]
        
        for num in nums:
            successive_accs.append(test_successive_estimation(2 ** n, n, SNRdB, num, test_size))
            our_accs.append(test_our_method(2 ** n, n, SNRdB, num, test_size))
        
        all_our_accs.append(our_accs)
        all_successive_accs.append(successive_accs)   
        all_kay_accs.append(kay_accs) 
    
    np.save('./data/kay_testing/t1_successive_est_accs_snr{}'.format(SNRdB), all_successive_accs)
    np.save('./data/kay_testing/t1_our_method_accs_snr{}'.format(SNRdB), all_our_accs)
    np.save('./data/kay_testing/t1_exps_used_snr{}'.format(SNRdB), exps)

def successive_est_params():
    exps = [8, 10, 12, 14, 16]
    SNRdB = 2

    all_successive_accs = []


    for n in exps:
        test_size = 20 * (2 ** n)
        successive_accs = []

        nums = [i for i in range(2,  n + 1)]

        if n <= 12:
            nums = [i for i in range(2,  2*n + 1)] # more if small n
    
        for num in nums:
            successive_accs.append(test_successive_estimation(2 ** n, n, SNRdB, num, test_size))
        all_successive_accs.append(successive_accs)   
    np.save('./data/kay_testing/successive_est_params_snr{}'.format(SNRdB), all_successive_accs)

def successive_est_params_big():
    exps = [20, 22, 24, 26, 28]
    SNRdB = 10

    all_successive_accs = []


    for n in exps:
        test_size = 50000
        successive_accs = []

        nums = [i for i in range(2,  n * n, 5)]

        if n <= 12:
            nums = [i for i in range(2,  2*n + 1)] # more if small n
    
        for num in nums:
            successive_accs.append(test_successive_estimation(2 ** n, n, SNRdB, num, test_size))
        all_successive_accs.append(successive_accs)   
    #np.save('./data/kay_testing/successive_est_params_snr{}'.format(SNRdB), all_successive_accs)

def kay_est_params_big():
    exps = [20, 22, 24, 26, 28]
    SNRdB = 10

    all_kay_accs = []


    for n in exps:
        test_size = 50000
        kay_accs = []

        nums = [i for i in range(2,  10)]

        if n <= 12:
            nums = [i for i in range(2,  2*n + 1)] # more if small n
    
        for num in nums:
            kay_accs.append(test_kays_method(2 ** n, n, SNRdB, num * n, test_size))
        print(kay_accs)
        all_kay_accs.append(kay_accs)   
    np.save('./data/kay_testing/kay_est_params_snr{}'.format(SNRdB), all_kay_accs)


def our_method_params():
    exps = [8, 10, 12, 14, 16]
    SNRdB = 2

    all_our_accs = []


    for n in exps:
        test_size = 8 * (2 ** n)
        our_accs = []

        nums = [i for i in range(1, n + 1)]


        for num in nums:
            our_accs.append(test_our_method(2 ** n, n, SNRdB, num, test_size))
        all_our_accs.append(our_accs)   
    print(all_our_accs)
    np.save('./data/kay_testing/our_method_params_snr{}'.format(SNRdB), all_our_accs)


def our_method_params_big():
    exps = [20, 22, 24, 26, 28]
    SNRdB = 20

    all_our_accs = []


    for n in exps:
        test_size = 50000
        our_accs = []

        nums = [i for i in range(1, n * n, 10)]
    
        for num in nums:
            our_accs.append(test_our_method(2 ** n, n, SNRdB, num, test_size))
        all_our_accs.append(our_accs)   
    np.save('./data/kay_testing/our_method_params_snr{}'.format(SNRdB), all_our_accs)


    
if __name__ == '__main__':    
    #our_method_params()
    successive_est_params_big()
    #our_method_params_big()
    #kay_est_params_big()
