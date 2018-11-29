import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import time

def make_signal(w,theta,m):
    """
    Assumes normalized amplitude
    """
    t = np.arange(m)
    signal = np.exp(1j*(w*t + theta))
    return signal

def make_noise(sigma2,m):
    noise_scaling = np.sqrt(sigma2/2)
    # noise is complex valued
    noise  = noise_scaling*np.random.randn(m) + 1j*noise_scaling*np.random.randn(m)
    return noise

def make_noisy_signal(w,theta,SNRdb,m):
    sigma2 = get_sigma2_from_snrdb(SNRdb)
    signal = make_signal(w,theta,m)
    noise  = make_noise(sigma2,m)
    return signal + noise

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

def make_batch_parity(signals, freqs):
    bits = [[(np.binary_repr(f, width=14).count('1') % 2)] for f in freqs]
    #bits = [[(int(np.binary_repr(f, width=14)[0]) % 2)] for f in freqs]
    return (signals, bits)

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

def test_kays(signals, freqs, N):
    count = 0
    for sig, freq in zip(signals, freqs):
        res = kays_method(sig)
        res = round(res * N / (2 * np.pi))
        if np.argmax(freq) == res:
            count += 1
    return count / len(signals)

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

def bit_to_freq(bits, N):
    possible = [i for i in range(N)]
    for b in bits:
        if not b[0]:
            possible = possible[:len(possible)//2]
        else:
            possible = possible[len(possible)//2:]
    return possible[0]

def convert_bits(bits):
    val = 0
    for i in range(len(bits)):
        val += bits[len(bits) - 1 - i] * (2**i)
    return val

def make_signal_random(w,theta, N, m, inds):
    sig = make_signal(w, theta, N)
    #chosen_indices = np.sort(np.random.choice(range(N), size=m, replace=False))
    return np.take(sig, inds)

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



# returns (accuracy, time) for nn forward prop
def test_nn(N, m, SNRdB, trials, num_iter, training_size, test_size):

    freqs_per_batch = 20
    training_size = (training_size // freqs_per_batch) * freqs_per_batch
    test_size = (test_size // freqs_per_batch) * freqs_per_batch
    layer = 2  

    log = int(np.log2(N))

    # Parameters
    learning_rate = 0.005
    batch_size = log * freqs_per_batch



    # Network Parameters
    num_classes = 1

    starts = [0] * 25 # if starting at 0 every resample


    indices = np.sort(np.random.choice(range(N), size=m, replace=False))
        

    t_bins = []
    t_freqs = []
    t_preds = []
    t_actuals = []
    all_times = []
        
    dict = {}
    for i in range(training_size // freqs_per_batch):
        #if i % 500 == 0:
        #    print('gen training batch:', i)
        batch_x, batch_y, batch_freqs, batch_rands = make_batch_noisy_lohi(batch_size // log, SNRdB - 2, N, m, indices, starts) # i % 3
        batch_x_pair = np.zeros((batch_size, m, 2))
        batch_x_pair[:, :, 0] = np.real(batch_x)
        batch_x_pair[:, :, 1] = np.imag(batch_x)
        dict[i] = (batch_x_pair, batch_y, batch_rands)
    test_dict = {}
    for i in range(test_size // freqs_per_batch):
        test_signals, test_freqs, freqs, test_rands = make_batch_noisy_lohi(batch_size // log, SNRdB, N, m, indices, starts)
        test_signals_pair = np.zeros((batch_size, m, 2))
        test_signals_pair[:, :, 0] = np.real(test_signals)
        test_signals_pair[:, :, 1] = np.imag(test_signals)
        test_dict[i] = (test_signals_pair, test_freqs, freqs, test_rands)
        
    for trial in range(trials):
        print('TRIAL:', trial)
        # tf Graph input
        X = tf.placeholder("float", [None, m, 2])
        Y = tf.placeholder("float", [None, num_classes])

        # weights for frequency classification
        weights = {i: tf.Variable(tf.random_normal([3, 2, 2])) for i in range(1, layer+1)} # increase out channels, less layers
        weights[0] = tf.Variable(tf.random_normal([5, 2, 2]))
        weights['out'] = tf.Variable(tf.random_normal([(m-4-(2*layer))*2, num_classes]))
        biases = {i: tf.Variable(tf.random_normal([2])) for i in range(layer+1)}
        biases['out'] = tf.Variable(tf.random_normal([num_classes]))


        def neural_net(x):
            layer_1 = tf.add(tf.nn.conv1d(x, weights[0], 1, 'VALID'), biases[0])
            hidden_1 = tf.nn.relu(layer_1)
            for i in range(1, layer+1):
                layer_1 = tf.add(tf.nn.conv1d(hidden_1, weights[i], 1, 'VALID'), biases[i]) # no padding
                hidden_1 = tf.nn.relu(layer_1) # try: elu, leaky
                ###hidden_1 = tf.layers.batch_normalization(hidden_1)
                ### instance normalize
            hidden_3 = tf.reshape(hidden_1, [batch_size, -1])
            out_layer = tf.matmul(hidden_3, weights['out']) + biases['out']
            return out_layer



        # Construct model
        logits = neural_net(X)
        prediction = tf.nn.sigmoid(logits)
        losses, accuracies = [], []

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=Y))  

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model
        pred_class = tf.greater(prediction, 0.5)
        correct_pred = tf.equal(pred_class, tf.equal(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)
            # print("Training Started")

            for step in range(1, num_iter + 1):
                batch_x_pair, batch_y, rands = dict[step % (training_size // freqs_per_batch)]

                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x_pair, Y: batch_y})
                
                if step % 500 == 0:
                    # Calculate batch loss and accuracy
                    loss, acc, pred = sess.run([loss_op, accuracy, prediction], feed_dict={X: batch_x_pair,
                                                                        Y: batch_y})                                           

                    accuracies.append(acc)
                    losses.append(loss)
                    #print("Freq Iter " + str(step) + ", Minibatch Loss= " + \
                    #    "{:.4f}".format(loss) + ", Training Accuracy= " + \
                    #    "{:.3f}".format(acc))
            # print("Training Finished")
            start = time.time()
            preds = [sess.run(prediction, feed_dict={X: test_dict[i][0], Y: test_dict[i][1]}) for i in range(len(test_dict))]  
            feedfwd_time = time.time() - start
            nn_acc = [sess.run(accuracy, feed_dict={X: test_dict[i][0], Y: test_dict[i][1]}) for i in range(len(test_dict))]   
            # print(nn_acc)
        t_preds.append(preds)
        t_actuals.append([test_dict[i][1] for i in range(len(test_dict))])

        t_bins.append(np.median(nn_acc))
        preds = np.round(preds)
        corr = []
        for a in range(len(test_dict)):
            fs = []
            for k in range(len(preds[a]) // log):
                    vals, timer = test_all_1bit_random(preds[a][k * log : (k+1) * log], N, m, test_dict[a][3][k], indices)
                    fs.append(vals)
                    feedfwd_time += timer
            corr.extend([fs[i] == test_dict[a][2][i] % N for i in range(len(fs))])
        t_freqs.append(np.sum(corr) / (len(fs) * len(test_dict)))
        all_times.append(feedfwd_time)
        print('accuracy:', t_freqs[-1], all_times[-1])
        
    print('N:', N, ' m:', m, max(t_freqs), all_times[np.argmax(t_freqs)])
    return max(t_freqs), all_times[np.argmax(t_freqs)]

accuracy_thresh = 0.95
trials = 10 
nn_trials = 3
nn_iters = 15000
nn_training_size = 10000
nn_test_size = 2500
SNRdB = 4
Ns = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

for N in Ns:
    print(N)
    times = []
    res = []
    prev_m = max(int(np.log2(N)), 9)  # otherwise negative dim after convolution
    for _ in range(trials):
        #full_signals = [make_signal(2*np.pi*i/N, 0, N) for i in range(N)]
        m = prev_m
        prev_good, prev_time = False, 0
        curr_good = False
        while not curr_good:

            result, timer = test_nn(N, m, SNRdB, nn_trials, nn_iters, nn_training_size, nn_test_size)
            print(m, result)
            trial_good = result >= accuracy_thresh
            curr_good = (trial_good != prev_good)
            if not curr_good and trial_good:
                m -= 1
            if not curr_good and not trial_good:
                m += 1
            if not curr_good:
                prev_time = time
                prev_good = trial_good
        prev_m = m if trial_good else m - 1
        res.append(prev_m)
        corr_time = timer if trial_good else prev_time
        print(prev_m, corr_time)
        times.append(corr_time / nn_test_size)
        prev_m = max(9, prev_m - 2)

    np.save('./data/nn_timing/times2_snr{}_accuracy{}_N{}'.format(SNRdB, accuracy_thresh, N), times)
    np.save('./data/nn_timing/ms_snr{}_accuracy{}_N{}'.format(SNRdB, accuracy_thresh, N), res)
    

        



