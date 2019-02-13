import numpy as np 
import matplotlib.pyplot as plt


# plots all dot products, corresponding to each index, vertial line at actual freq
def plot_mle(dots, inds, freq, title=''):
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    ax.plot([i for i in inds], dots,'ro')
    ax.plot([freq], [dots[np.where(inds == freq)]], 'go')
    ax.set_xlabel('Frequency') 
    ax.set_ylabel('Dot product magnitude with signal')
    ax.set_title(title)
    ax.axvline(x=freq)
    plt.show()
    print(inds[np.argmax(dots)] == freq) # was mle correct?

# gets percentage of misclassified that were of each freq
def plot_missed_freqs(base, exp, wrongs, title=''):
    N = (base ** exp)
    hist_list = [0] * N
    for i in wrongs:
        hist_list[i] += 1
    hist_list = [i / len(wrongs) for i in hist_list]
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Frequency') 
    ax.set_ylabel('Percent of all misclassified or P(freq | misclassified)')
    ax.set_xlim(0, N)
    ax.plot(list(range(N)), hist_list, 'go')
    ax.set_title(title)
    for i in range(4):
        for k in range((2 ** i)):
            ax.axvline(x = k * (N // (2 ** i)), color='red')
    max_inds = np.argpartition(np.array(hist_list), -10)[-10:]
    max_missed = [(max_inds[i], round(hist_list[max_inds[i]], 4)) for i in range(len(max_inds))]
    max_missed.sort(key=lambda x: -x[1])
    plt.show()
    print(max_missed)

# gets percentage of each freq that were misclassified
def plot_missed_freqs_total_percent(base, exp, wrongs, freq_counts, title=''):
    N = (base ** exp)
    hist_list = [0] * N
    for i in wrongs:
        hist_list[i] += 1
    hist_list = [hist_list[i] / freq_counts[i] for i in range(len(hist_list))]
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Frequency') 
    ax.set_ylabel('Liklihood of being misclassified or P(misclassified | freq)')
    ax.set_xlim(0, N)
    ax.plot(list(range(N)), hist_list, 'go')
    ax.set_title(title)
    for i in range(4):
        for k in range((2 ** i)):
            ax.axvline(x = k * (N // (2 ** i)), color='red')
    max_inds = np.argpartition(np.array(hist_list), -10)[-10:]
    max_missed = [(max_inds[i], round(hist_list[max_inds[i]], 4)) for i in range(len(max_inds))]
    max_missed.sort(key=lambda x: -x[1])
    plt.show()
    print(max_missed)

# gets avg bits off that misclassified of each freq were
def plot_bits_off(base, exp, wrongs, bits_off, avg_func=np.mean, title=''):
    N = (base ** exp)
    bits_missed = [[] for i in range(N)]
    for freq, bits in zip(wrongs, bits_off):
        bits_missed[freq].append(bits)
    avg_bits_missed = [avg_func(i) if len(i) > 0 else 0 for i in bits_missed]
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Frequency') 
    ax.set_ylabel('Bits off from correct frequency of all misclassified')
    ax.set_xlim(0, N)
    ax.plot(list(range(N)), avg_bits_missed, 'go')
    ax.set_title(title)
    for i in range(4):
        for k in range((2 ** i)):
            ax.axvline(x = k * (N // (2 ** i)), color='red')
    plt.show()

# get histogram of how much each bit was missed for a particular freq
def plot_missed_bits_for_freq(base, exp, freq, wrongs, wrong_bits, title=''):
    bits = []
    for wrong_freq, wrong_b in zip(wrongs, wrong_bits):
        if wrong_freq == freq:
            bits.extend(wrong_b)
    fig = plt.figure(figsize=(15,4))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Bit index') 
    ax.set_ylabel('Number misclassified')
    ax.set_title(title)
    ax.hist(bits, bins=[i for i in range(exp)]) 
    plt.show()



