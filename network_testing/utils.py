import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from functools import reduce
import time

## SIGNAL UTILS

def get_sigma2_from_snrdb(SNR_db):
    return 10**(-SNR_db/10)

def make_signal(w, theta, m):
    """
    Assumes normalized amplitude
    """
    t = np.arange(m)
    signal = np.exp(1j*(w*t + theta))
    return signal

def make_noise(sigma2, m):
    noise_scaling = np.sqrt(sigma2/2)
    # noise is complex valued
    noise  = noise_scaling*np.random.randn(m) + 1j*noise_scaling*np.random.randn(m)
    return noise

def make_noisy_signal(w,theta,SNRdb,m):
    sigma2 = get_sigma2_from_snrdb(SNRdb)
    signal = make_signal(w,theta,m)
    noise  = make_noise(sigma2,m)
    return signal + noise

def make_signal_random(w,theta, N, m, inds):
    sig = make_signal(w, theta, N)
    #chosen_indices = np.sort(np.random.choice(range(N), size=m, replace=False))
    return np.take(sig, inds)

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

## GENERAL UTILS

def hamming(pred, act):
    pred = np.array(pred)
    act = np.array(act)
    return np.count_nonzero(pred != act)

def make_binary(freqs, N):
    w = math.ceil(np.log2(N))
    return [[int(a) for a in list(np.binary_repr(f, width=w))] for f in freqs] 

def convert_bits(bits, base):
    val = 0
    for i in range(len(bits)):
        val += bits[len(bits) - 1 - i] * (base**i)
    return val

def convert_int_to_bits(num, base, exp):
    bits_base = np.base_repr(num, base)
    bits_base = [int(a) for a in bits_base]
    bits_base = [0] * (exp - len(bits_base)) + bits_base 
    return bits_base

def get_bits_for_int(num, bases, exps):
    first = convert_int_to_bits(num % (bases[0] ** exps[0]), bases[0], exps[0])
    second = first = convert_int_to_bits(num % (bases[1] ** exps[1]), bases[1], exps[1])
    return [first, second]

def get_hamming(freq_bits, pred_bits):
    return [hamming(freq_bits[0], pred_bits[0]), hamming(freq_bits[1], pred_bits[1])]

def one_hot(N, batch_size, freqs):
    freqs_one_hot = np.zeros((batch_size, N))
    freqs_one_hot[np.arange(batch_size), freqs] = 1
    return freqs_one_hot

def chinese_remainder(n, a):
    sum = 0
    prod = reduce(lambda a, b: a*b, n)
    for n_i, a_i in zip(n, a):
        p = prod / n_i
        sum += a_i * mul_inv(p, n_i) * p
    return int(sum % prod)
 
 
 
def mul_inv(a, b):
    b0 = b
    x0, x1 = 0, 1
    if b == 1: return 1
    while a > 1:
        q = a // b
        a, b = b, a%b
        x0, x1 = x1 - q * x0, x0
    if x1 < 0: x1 += b0
    return x1


## KAY UTILITIES

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

def test_kays(signals, freqs, N):
    count = 0
    for sig, freq in zip(signals, freqs):
        res = kays_method(sig)
        res = round(res * N / (2 * np.pi))
        if np.argmax(freq) == res:
            count += 1
    return count / len(signals)

## SINGLETON UTILITIES

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

## MLE UTILITIES

def test_mle(signals, freqs, N, m):
    count = 0
    for sig, freq in zip(signals, freqs):
        cleans = [make_signal(np.pi * 2 * w / N, 0, m) for w in range(N)]
        dots = [np.absolute(np.vdot(sig, clean)) for clean in cleans]
        if np.argmax(dots) == np.argmax(freq):
            count += 1
    return count / len(signals)

def test_noisy_mle(N, m, signals, freqs):
    count = 0  
    '''imag_signals = []
    for index in range(len(signals)):
        sig = signals[index]
        imag_sig = [(sig[i] + 1j*sig[i+1]) for i in np.arange(len(sig), step=2)]
        imag_signals.append(imag_sig)'''
    cleans = [make_signal(2*np.pi*i/N, 0, m) for i in range(N)]
                     
    for index in range(len(signals)):
        dots = [np.absolute(np.vdot(signals[index], cleans[i])) for i in range(N)]
        if np.argmax(freqs[index]) == np.argmax(dots):
            #print(np.argmax(dots))
            count += 1
    return count / len(freqs)

## NN UTILITIES

# inds=random indices used to test mle
def make_noisy_lohi(SNRdb, N, m, freq, inds, starts):
    signals, vals = [], []
    steps = int(np.log2(N))
    w = (2 * np.pi * freq / N) % (2 * np.pi)
    sig = make_noisy_signal(w, 0, SNRdb, N)
    #start = 0
    for i in range(int(np.log2(N))):
        #start = start + np.random.randint(N // 4) if i > 0 else 0
        signals.append([sig[(starts[i] + a * (2**i)) % N] for a in range(m)])
        if (freq * (2**i)) % (N) < N / 2:
            vals.append([0])
        else:
            vals.append([1])
    return signals, vals, np.take(sig, inds)

# for general base
# inds=random indices used to test mle
def make_noisy_lohi2(SNRdb, N, m, freq, inds, starts, base):
    signals, vals = [], []
    steps = int(np.log(N) / np.log(base))
    w = (2 * np.pi * freq / N) % (2 * np.pi)
    sig = make_noisy_signal(w, 0, SNRdb, N)
    #start = 0
    for i in range(steps):
        #start = start + np.random.randint(N // 4) if i > 0 else 0
        signals.append([sig[(starts[i] + a * (base**i)) % N] for a in range(m)])
        freq_array = [0] * base
        freq_array[((freq * (base**i) % N) * base) // N] = 1
        vals.append(freq_array)
    return signals, vals, np.take(sig, inds)

# N = divisor of w0
# m = num samples
# starts = shift for each subsequent sample
def make_batch_noisy_lohi(batch_size, SNRdb, N, m, inds, starts=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
    freqs = []
    randoms = []
    freqs.append(np.random.randint(0, N))
    test_signals, test_freqs, test_rand = make_noisy_lohi(SNRdB, N, m, freqs[-1], inds, starts)
    randoms.append(test_rand)
    for i in range(1, batch_size):
        freqs.append(np.random.randint(0, N))
        a, b, c = make_noisy_lohi(SNRdB, N, m, freqs[-1], inds, starts)
        test_signals.extend(a)
        test_freqs.extend(b)
        randoms.append(c)
    return test_signals, test_freqs, freqs, randoms

# for general base
# N = divisor of w0
# m = num samples
# starts = shift for each subsequent sample
def make_batch_noisy_lohi(batch_size, SNRdb, N, m, inds, base, starts=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
    freqs = []
    randoms = []
    freqs.append(np.random.randint(0, N))
    test_signals, test_freqs, test_rand = make_noisy_lohi2(SNRdB, N, m, freqs[-1], inds, starts, base)
    randoms.append(test_rand)
    for i in range(1, batch_size):
        freqs.append(np.random.randint(0, N))
        a, b, c = make_noisy_lohi2(SNRdB, N, m, freqs[-1], inds, starts, base)
        test_signals.extend(a)
        test_freqs.extend(b)
        randoms.append(c)
    return test_signals, test_freqs, freqs, randoms

def test_all_1bit(bits, N, m, signal):
    vals = []
    new_signal = np.zeros((m))
    new_signal = signal[:, 0] + 1j*signal[:, 1]
    signal = new_signal
    min_ind = convert_bits(bits)
    sig = make_signal((convert_bits(bits) * 2*np.pi/N), 0, m)
    min_val = np.vdot(sig - signal, sig - signal)
    vals.append(min_val)
    for i in range((len(bits) // 2)): # because less significant bits have more wiggle - easier to cause false error
        bits[i] = abs(bits[i] - 1)
        sig = make_signal((convert_bits(bits) * 2*np.pi/N), 0, m)
        resid = np.vdot(sig - signal, sig - signal)
        vals.append(resid)
        if resid < min_val: 
            min_val = resid
            min_ind = convert_bits(bits)
        bits[i] = abs(bits[i] - 1)
    #print('thresh: ', min_ind)
    #print(vals)
    #print(min_ind == convert_bits(bits))
    return min_ind

def test_all_1bit_random(bits, N, m, signal, inds): # try set of all combined measurements mle
    vals = []
    total_time = 0
    #new_signal = np.zeros((m))
    #new_signal = signal[:, 0] + 1j*signal[:, 1]
    #signal = new_signal
    min_ind = convert_bits(bits)
    sig = make_signal_random((convert_bits(bits) * 2*np.pi/N), 0, N, m, inds)
    t_start = time.time()
    min_val = np.absolute(np.vdot(sig, signal))
    total_time += (time.time() - t_start)
    vals.append(min_val)
    for i in range((len(bits))): # because less significant bits have more wiggle - easier to cause false error
        bits[i] = abs(bits[i] - 1)
        sig = make_signal_random((convert_bits(bits) * 2*np.pi/N), 0, N, m, inds)
        t_start = time.time()
        resid = np.absolute(np.vdot(sig, signal))
        vals.append(resid)
        if resid > min_val: 
            min_val = resid
            min_ind = convert_bits(bits)
        total_time += (time.time() - t_start)
        bits[i] = abs(bits[i] - 1)
    #print('thresh: ', min_ind)
    #print(vals)
    #print(min_ind == convert_bits(bits))
    return min_ind, total_time

# want to generate train, test data
def generate_data_dicts(N, ms, bases, exps, dict_size, batch_size, SNRdB, indices):
    dict1, dict2 = {}, {}
    inds1, inds2 = [], []
    sigma2 = get_sigma2_from_snrdb(SNRdB)
    for i in range(exps[0]):
        inds1.append([(k * (bases[1] ** exps[1]) * (bases[0] ** i)) % N for k in range(ms[0])])
    for i in range(exps[1]):
        inds2.append([(k * (bases[0] ** exps[0]) * (bases[1] ** i)) % N for k in range(ms[1])])
    for i in range(dict_size):
        sigs1, sigs2 = [], []
        freqs1, freqs2 = [], []
        randoms, orig_freqs = [], []
        for k in range(batch_size):
            curr_freq = np.random.randint(0, N)
            w = (2 * np.pi * curr_freq / N) % (2 * np.pi)
            orig_freqs.append(curr_freq)
            signal_values = {}
            for j in range(len(inds1)):
                for ind in inds1[j]:
                    if ind not in signal_values:
                        signal_values[ind] = np.exp(1j*(w*ind)) + make_noise(sigma2, 1)[0]
                sigs1.append([signal_values[ind] for ind in inds1[j]])
                freq_one_hot = [0] * bases[0]
                freq_one_hot[((curr_freq * (bases[1]**exps[1]) * (bases[0] ** j) % N) * bases[0]) // N] = 1
                freqs1.append(freq_one_hot)
                
            for j in range(len(inds2)):
                for ind in inds2[j]:
                    if ind not in signal_values:
                        signal_values[ind] = np.exp(1j*(w*ind)) + make_noise(sigma2, 1)[0]
                sigs2.append([signal_values[ind] for ind in inds2[j]])
                freq_one_hot = [0] * bases[1]
                freq_one_hot[((curr_freq * (bases[0]**exps[0]) * (bases[1] ** j) % N) * bases[1]) // N] = 1
                freqs2.append(freq_one_hot)
                
            for ind in indices:
                if ind not in signal_values:
                    signal_values[ind] = np.exp(1j*(w*ind)) + make_noise(sigma2, 1)[0]
            randoms.append([signal_values[ind] for ind in indices])
        dict1[i] = (sigs1, freqs1, randoms, orig_freqs)
        dict2[i] = (sigs2, freqs2, randoms, orig_freqs)
    return (dict1, dict2)
            

def train_nn(N, m, train_dict, batch_size, dict_size, num_classes=2, layer=3, num_iter=10000, learning_rate=0.005):
    tf.reset_default_graph()
    
    X = tf.placeholder("float", [None, m, 2], name = 'X')
    Y = tf.placeholder("float", [None, num_classes], name = 'Y')

    # weights for frequency classification
    weights = {i: tf.Variable(tf.random_normal([3, 2, 2]), name='w{}'.format(i)) for i in range(1, layer+1)} # increase out channels, less layers
    weights[0] = tf.Variable(tf.random_normal([5, 2, 2]), name='w0')
    weights['out'] = tf.Variable(tf.random_normal([(m - 4 - (2 * layer)) * 2, num_classes]), name='wout')
    biases = {i: tf.Variable(tf.random_normal([2]), name='b{}'.format(i)) for i in range(layer+1)}
    biases['out'] = tf.Variable(tf.random_normal([num_classes]), name='bout')
    


    def neural_net(x):
        layer_1 = tf.add(tf.nn.conv1d(x, weights[0], 1, 'VALID'), biases[0])
        hidden_1 = tf.nn.relu(layer_1)
        for i in range(1, layer+1):
            layer_1 = tf.add(tf.nn.conv1d(hidden_1, weights[i], 1, 'VALID'), biases[i]) # no padding
            hidden_1 = tf.nn.relu(layer_1) # try: elu, leaky
            ###hidden_1 = tf.layers.batch_normalizationf(hidden_1)
            ### instance normalize
        hidden_3 = tf.reshape(hidden_1, [batch_size, -1])
        out_layer = tf.matmul(hidden_3, weights['out']) + biases['out']
        return out_layer



    # Construct model
    
    logits = neural_net(X)
    prediction = tf.nn.softmax(logits)
    losses, accuracies = [], []

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))  

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    pred_class = tf.argmax(prediction, 1)
    tf.identity(pred_class, name="pred_class_op")
    correct_pred = tf.equal(pred_class, tf.argmax(Y, 1))
    tf.identity(correct_pred, name="correct_pred_op")
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    saver = tf.train.Saver()

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        # print("Training Started")

        for step in range(1, num_iter + 1):
            batch_x, batch_y, rands, freqs = train_dict[step % dict_size]
            batch_x_pair = np.zeros((batch_size, m, 2))
            batch_x_pair[:, :, 0] = np.real(batch_x)
            batch_x_pair[:, :, 1] = np.imag(batch_x)

            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x_pair, Y: batch_y})

            if step % 500 == 0:
                # Calculate batch loss and accuracy
                loss, acc, pred = sess.run([loss_op, accuracy, prediction], feed_dict={X: batch_x_pair,
                                                                    Y: batch_y})                                           

                accuracies.append(acc)
                losses.append(loss)
                print("Freq Iter " + str(step) + ", Minibatch Loss= " + \
                    "{:.4f}".format(loss) + ", Training Accuracy= " + \
                    "{:.3f}".format(acc))
        print("Training Finished")
        #saver.save(sess, './N:{}_m:{}_base:{}'.format(N, m, num_classes))
        saver.save(sess, './N{}m{}base{}'.format(N, m, num_classes))
    print('done')
    
        
def test_nn(N, m, train_dict, dict_size, batch_size, base, exp):
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('N{}m{}base{}.meta'.format(N, m, base)) # fix to actual checkpoint formatted above
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        X = tf.get_default_graph().get_tensor_by_name("X:0")
        Y = tf.get_default_graph().get_tensor_by_name("Y:0")
        pred_class = tf.get_default_graph().get_tensor_by_name("pred_class_op:0")
        all_preds = []
        all_actuals = []
        total_time = 0
        for i in range(dict_size):
            batch_x, batch_y, rands, freqs = train_dict[i]
            batch_x_pair = np.zeros((batch_size, m, 2))
            batch_x_pair[:, :, 0] = np.real(batch_x)
            batch_x_pair[:, :, 1] = np.imag(batch_x)
            start_time = time.time()
            preds = sess.run(pred_class, feed_dict={X: batch_x_pair, Y: batch_y})
            end_time = time.time()
            total_time += end_time - start_time
            acts = np.array([np.argmax(a) for a in batch_y])
            #if list(preds) != [np.argmax(a) for a in batch_y]:
            #print(preds, np.array([np.argmax(a) for a in batch_y]))
            pred_vals = [convert_bits(preds[exp * i : (exp + 1) * i], base) for i in range(batch_size // exp)]
            actual_vals = [convert_bits(acts[exp * i : (exp + 1) * i], base) for i in range(batch_size // exp)]
            #print(pred_vals, actual_vals)
            all_preds.append(preds)
            all_actuals.append(acts)
            #else:
            #    print('good')
        
        
    return (total_time, all_preds, all_actuals)
    
    # do timing and accuracy tests
    
def determine_freq(N, bases, exps, freqs1, freqs2):
    first = convert_bits(freqs1, bases[0])
    second = convert_bits(freqs2, bases[1])
    guess = chinese_remainder([bases[0] ** exps[0], bases[1] ** exps[1]], [first, second])
    return (0, guess, (freqs1, freqs2))

def determine_freq_full(N, bases, exps, freqs1, freqs2, rands, indices):
    all_firsts = set()
    all_firsts.add(convert_bits(freqs1, bases[0]))
    for i in range(exps[0]):
        bits = np.copy(freqs1)
        for j in range(bases[0]):
            bits[i] = j
            all_firsts.add(convert_bits(bits, bases[0]))
    firsts = list(all_firsts) * ((bases[1] - 1) * exps[1] + 1)
    all_seconds = set()
    all_seconds.add(convert_bits(freqs2, bases[1]))
    for i in range(exps[1]):
        bits = np.copy(freqs2)
        for j in range(bases[1]):
            bits[i] = j
            all_seconds.add(convert_bits(bits, bases[1]))
    seconds = []
    for i in all_seconds:
        seconds.extend([i] * ((bases[0] - 1) * exps[0] + 1))
    max_dot, best_freq = 0, 0
    total_time = 0
    for first, second in zip(firsts, seconds):
        freq = chinese_remainder([bases[0] ** exps[0], bases[1] ** exps[1]], [first, second])
        w = 2 * np.pi * freq / N
        random_sig = [np.exp(1j*(w*t)) for t in indices]
        t_start = time.time()
        dot_product = np.absolute(np.vdot(random_sig, rands))
        if dot_product > max_dot:
            max_dot = dot_product
            best_freq = freq
        t_end = time.time()
        total_time += t_end - t_start
    bits_1 = convert_int_to_bits(best_freq % (bases[0] ** exps[0]), bases[0], exps[0])
    bits_2 = convert_int_to_bits(best_freq % (bases[1] ** exps[1]), bases[1], exps[1])
    return (total_time, best_freq, (bits_1, bits_2))
        
    
    
def get_final_frequency(N, train_dict_1, train_dict_2, dict_size, batch_size, bases, exps, indices, all_preds, all_acts, verbose=False):
    all_predictions = {} # (freq, (bits1, bits2)) for each batch for no error correcting 
    all_predictions_full = {} # (freq, (bits1, bits2)) for each batch for full 1 bit error correcting
    total_time, total_time_full = 0, 0
    for i in range(dict_size):
        batch_x1, batch_y1, rands1, freqs1 = train_dict_1[i]
        batch_x2, batch_y2, rands2, freqs2 = train_dict_2[i]
        predictions, predictions_full = [], []
        for j in range(batch_size):
            pred = determine_freq(N, bases, exps, all_preds[0][i][exps[0] * j : exps[0] * (j+1)], all_preds[1][i][exps[1] * j : exps[1] * (j+1)])
            pred_full = determine_freq_full(N, bases, exps, all_preds[0][i][exps[0] * j : exps[0] * (j+1)], all_preds[1][i][exps[1] * j : exps[1] * (j+1)], rands1[j], indices)
            predictions.append(pred[1:])
            predictions_full.append(pred_full[1:])
            total_time += pred[0]
            total_time_full += pred_full[0]
        #predictions = [determine_freq(N, bases, exps, all_preds[0][i][exps[0] * j : exps[0] * (j+1)], all_preds[1][i][exps[1] * j : exps[1] * (j+1)]) for j in range(batch_size)]
        #predictions_full = [determine_freq_full(N, bases, exps, all_preds[0][i][exps[0] * j : exps[0] * (j+1)], all_preds[1][i][exps[1] * j : exps[1] * (j+1)], rands1[j], indices) for j in range(batch_size)]
        
        
        #determine_freq(N, bases, exps, all_acts[0][i][exps[0] * j : exps[0] * (j+1)], all_acts[1][i][exps[1] * j : exps[1] * (j+1)])
        if verbose:
            for j in range(batch_size):
                if predictions[j][0] != freqs1[j] and predictions_full[j][0] != freqs1[j]:
                    print('guess (small):', predictions[j][0], 'guess (full):', predictions_full[j][0], 'actual:', freqs1[j])
                elif predictions_full[j][0] != freqs1[j]:
                    print('guess (full):', predictions_full[j][0], 'actual:', freqs1[j])
                elif predictions[j][0] != freqs1[j]:
                    print('guess (small):', predictions[j][0], 'actual:', freqs1[j])
        all_predictions[i] = predictions
        all_predictions_full[i] = predictions_full
            
    ## add all times together
    print('done')
    return (all_predictions, total_time), (all_predictions_full, total_time_full)
        
    
# returns (test_dict_1, test_dict_2), which are of size dict_sizes[0] and have at each index: batch_x, batch_y, rands, freqs
#         (all_predictions, total_time), which is of size dict_sizes[1] and have at each index: freq, (bits_1, bits_2), and then the total time by method 1 (no error correcting)
#         (all_predictions_full, total_time_full), which is same as above for method 2 (1 bit comprehensive error correcting using mle)
def frequency_detection(ms, bases, exps, dict_sizes, batch_size, SNRdB, num_iters=5000, layers=3):
    tf.reset_default_graph()
    N = (bases[0] ** exps[0]) * (bases[1] ** exps[1])
    indices = np.sort(np.random.choice(range(N), size=ms[0], replace=False))
    d1, d2 = generate_data_dicts(N, ms, bases, exps, dict_sizes[0], batch_size, SNRdB, indices)
    t1, t2 = generate_data_dicts(N, ms, bases, exps, dict_sizes[1], batch_size, SNRdB, indices)
    train_nn(N, ms[0], d1, batch_size * exps[0], dict_sizes[0], num_classes=bases[0], layer=layers, num_iter=num_iters)
    time_feedfwd1, allp1, alla1 = test_nn(N, ms[0], t1, dict_sizes[1], batch_size * exps[0], bases[0], exps[0])
    train_nn(N, ms[1], d2, batch_size * exps[1], dict_sizes[0], num_classes=bases[1], layer=layers, num_iter=num_iters)
    time_feedfwd2, allp2, alla2 = test_nn(N, ms[1], t2, dict_sizes[1], batch_size * exps[1], bases[1], exps[1])
    (all_predictions, total_time), (all_predictions_full, total_time_full) = get_final_frequency(N, t1, t2, dict_sizes[1], batch_size, bases, exps, indices, [allp1, allp2], [alla1, alla2])
    return (t1, t2), (all_predictions, total_time + time_feedfwd1 + time_feedfwd2), (all_predictions_full, total_time_full + time_feedfwd1 + time_feedfwd2)

def calculate_accuracy(t1, all_preds, dict_size, batch_size):
    correct = 0
    for i in range(dict_size):
        batch_x, batch_y, rands, freqs = t1[i]
        for j in range(batch_size):
            if all_preds[i][j][0] == freqs[j]:
                correct += 1
    print(correct / (dict_size * batch_size))
    return correct / (dict_size * batch_size)

    
