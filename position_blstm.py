'''
Build a Wear parts Identification lstm
'''

from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import conv as sconv
from theano.tensor.signal import pool as spool
from theano.tensor.nnet import conv as nconv 
from theano.tensor.nnet import sigmoid as nsigmoid 
import datapro

datasets = {'datapro': (datapro.load_data)}

SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """
    n = n - (n%minibatch_size)
    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def get_dataset(name):
    return datasets[name]

def get_layer(name):
    fns = layers[name]
    return fns

def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)

def init_params(options):
    params = OrderedDict()
  
    #cnn 
    if options['use_cnn'] == True:
    	params = get_layer('cnn')[0](options,
             	                     params,
                	             prefix='cnn')
        # embedding
        params['cnn_lstm'] = (0.01 * numpy.random.rand(options['cnn_kernel_num'])).astype(config.floatX)
 

    #before lstm
    params['before_lstm'] = (0.01 * numpy.random.rand(options['data_x_length'], options['dim_proj'])).astype(config.floatX)
  
 
    #lstm
    params = get_layer('lstm')[0](options,
                                  params,
                                  prefix='lstm')

    #dnn
    if options['use_dnn'] == True:
        params = get_layer('dnn')[0](options,
                                     params,
                                     prefix='dnn') 
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                                options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)
    
     
    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_cnn(options, params, prefix='cnn'):
    randn1 = numpy.random.rand(options['cnn_kernel_num'], options['cnn_kernel_length'])
    W1 = (0.01 * randn1).astype(config.floatX)
    params[_p(prefix, 'W1')] = W1
    if options['cnn_layer_num'] == 2:
        randn =  numpy.random.rand(options['cnn_kernel_num'], options['cnn_kernel_num'], options['cnn_kernel_length'])
        W2 = (0.01 * randn).astype(config.floatX)
        params[_p(prefix, 'W2')] = W2

    b = numpy.zeros((options['cnn_layer_num'], options['cnn_kernel_num'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params

def cnn_layer(tparams, state_below, options, prefix='cnn'):
    klength = options['cnn_kernel_length']
    for i in range(options['cnn_layer_num']):
        if i==0:
            cnn_out = sconv.conv2d(state_below,
                                   tparams[_p(prefix, 'W1')].dimshuffle([0,1,'x']))
            cnn_out = cnn_out + tparams[_p(prefix, 'b')][i].dimshuffle(['x',0,'x','x'])
            cnn_out = tensor.nnet.relu(cnn_out)
            #cnn_out = nsigmoid(cnn_out)
            if options['use_pooling'] == True:
                cnn_out = spool.pool_2d(cnn_out, (3,1), ignore_border=False,  mode=options['pool_mode']).astype(config.floatX)
        if i!=0:
            cnn_out = nconv.conv2d(cnn_out,
                                   tparams[_p(prefix, 'W2')].dimshuffle([0,1,2,'x']))
            cnn_out = cnn_out + tparams[_p(prefix, 'b')][i].dimshuffle(['x',0,'x','x'])
            cnn_out = tensor.nnet.relu(cnn_out)
            #cnn_out = nsigmoid(cnn_out)
    return cnn_out.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    if options['use_blstm']:
        bW = numpy.concatenate([ortho_weight(options['dim_proj']),
                                ortho_weight(options['dim_proj']),
                                ortho_weight(options['dim_proj']),
                                ortho_weight(options['dim_proj'])], axis=1)
        params[_p(prefix, 'bW')] = bW
        bU = numpy.concatenate([ortho_weight(options['dim_proj']),
                                ortho_weight(options['dim_proj']),
                                ortho_weight(options['dim_proj']),
                                ortho_weight(options['dim_proj'])], axis=1)
        params[_p(prefix, 'bU')] = bU
        bb = numpy.zeros((4 * options['dim_proj'],))
        params[_p(prefix, 'bb')] = bb.astype(config.floatX)

        
    if options['lstm_layer_num'] == 2:
        W2 = numpy.concatenate([ortho_weight(options['dim_proj']),
                                ortho_weight(options['dim_proj']),
                                ortho_weight(options['dim_proj']),
                                ortho_weight(options['dim_proj'])], axis=1)
        params[_p(prefix, 'W2')] = W2
        U2 = numpy.concatenate([ortho_weight(options['dim_proj']),
                                ortho_weight(options['dim_proj']),
                                ortho_weight(options['dim_proj']),
                                ortho_weight(options['dim_proj'])], axis=1)
        params[_p(prefix, 'U2')] = U2
        b2 = numpy.zeros((4 * options['dim_proj'],))
        params[_p(prefix, 'b2')] = b2.astype(config.floatX)

        if options['use_blstm']:
            bW2 = numpy.concatenate([ortho_weight(options['dim_proj']),
                                     ortho_weight(options['dim_proj']),
                                     ortho_weight(options['dim_proj']),
                                     ortho_weight(options['dim_proj'])], axis=1)
            params[_p(prefix, 'bW2')] = bW2
            bU2 = numpy.concatenate([ortho_weight(options['dim_proj']),
                                     ortho_weight(options['dim_proj']),
                                     ortho_weight(options['dim_proj']),
                                     ortho_weight(options['dim_proj'])], axis=1)
            params[_p(prefix, 'bU2')] = bU2
            bb2 = numpy.zeros((4 * options['dim_proj'],))
            params[_p(prefix, 'bb2')] = bb2.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, W, b, U, prefix='lstm', layername='_layer'):

    nsteps = state_below.shape[0] 
    n_samples = state_below.shape[1]


    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(x_, h_, c_):
        preact = tensor.dot(h_, U)
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c

        h = o * tensor.tanh(c)

        return h, c
    
    state_below = (tensor.dot(state_below, W) +
                   b)
    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, layername),
                                n_steps=nsteps)

    return rval[0]



def param_init_dnn(options, params, prefix='dnn'):
    W1 = (0.01 * numpy.random.rand(options['dim_proj'], options['dnn_layer1_num'])).astype(config.floatX)
    params[_p(prefix, 'W1')] = W1
    b1 = numpy.zeros((options['dnn_layer1_num'],))
    params[_p(prefix, 'b1')] = b1.astype(config.floatX)

    W2 = (0.01 * numpy.random.rand(options['dnn_layer1_num'], options['dnn_layer2_num'])).astype(config.floatX)
    params[_p(prefix, 'W2')] = W2
    b2 = numpy.zeros((options['dnn_layer2_num'],))
    params[_p(prefix, 'b2')] = b2.astype(config.floatX)

    return params

def dnn_layer(tparams, state_below, options, prefix='dnn'):  
    cnn_out = tensor.nnet.sigmoid(tensor.dot(state_below, tparams[_p(prefix, 'W1')]) + tparams[_p(prefix, 'b1')])
    cnn_out = tensor.nnet.sigmoid(tensor.dot(cnn_out, tparams[_p(prefix, 'W2')]) + tparams[_p(prefix, 'b2')])
 
    return cnn_out.astype(config.floatX)

layers = {'cnn': (param_init_cnn, cnn_layer),
          'lstm': (param_init_lstm, lstm_layer),
          'dnn': (param_init_dnn, dnn_layer)}

def adadelta(lr, tparams, grads, x, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared',
                                    allow_input_downcast=True)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

def sgd(lr, tparams, grads, x, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared',
                                    allow_input_downcast=True)

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update

def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.tensor3('x', dtype= config.floatX)
    y = tensor.vector('y', dtype='int32')

    n_timesteps = options['maxlen']
    n_samples = options['batch_size']

    if options['use_cnn'] == True:
        proj = get_layer('cnn')[1](tparams, x, options, prefix='cnn')
        proj = tensor.dot(proj.transpose((0,2,3,1)), tparams['cnn_lstm'])
        proj = tensor.dot(proj, tparams['before_lstm'])
    else:
        proj = tensor.dot(x, tparams['before_lstm'])

    proj = proj.transpose((1,0,2))
    
    proj1= get_layer('lstm')[1](tparams, proj, options, tparams['lstm_W'], 
                               tparams['lstm_b'], tparams['lstm_U'], prefix='lstm', layername='layer1')
    if options['use_blstm']: 
        proj1 = proj1[::-1] + get_layer('lstm')[1](tparams, proj[::-1], options, tparams['lstm_bW'],
                                                 tparams['lstm_bb'], tparams['lstm_bU'], prefix='lstm', layername='layer1b')
        proj1 = proj1[::-1] 
    if options['lstm_layer_num'] == 2:
        proj2= get_layer('lstm')[1](tparams, proj1, options, tparams['lstm_W2'], 
                                   tparams['lstm_b2'], tparams['lstm_U2'], prefix='lstm', layername='layer2')
        if options['use_blstm']:
            proj2 = proj2[::-1] +  get_layer('lstm')[1](tparams, proj1[::-1], options, tparams['lstm_bW2'],
                                                      tparams['lstm_bb2'], tparams['lstm_bU2'], prefix='lstm', layername='layer2b')
            proj2 = proj2[::-1]
    if options['lstm_layer_num'] == 1:
        proj = proj1
    if options['lstm_layer_num'] == 2:
        proj = proj2

    current_steps = proj.shape[0].astype(config.floatX)
    proj = proj.sum(axis=0)
    proj = proj / current_steps
    
    if options['use_dnn']:
        proj = get_layer('dnn')[1](tparams, proj, options, prefix='dnn')

    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)
  
    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x], pred, name='f_pred_prob', allow_input_downcast=True)
    f_pred = theano.function([x], pred.argmax(axis=1), name='f_pred', allow_input_downcast=True)
    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, y, f_pred_prob, f_pred, cost
def pred_probs(f_pred_prob, data, iterator, verbose=False):
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x = [[data[0][t] for t in valid_index]]
        y = numpy.array(data[1])[valid_index]
                                
        pred_probs = f_pred_prob(x)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_error(f_pred, data, iterator, pr=False, resul_tname=''):
    valid_err = 0
    x = []
    if pr==True: 
        f = open(result_name, 'a')
    for _, valid_index in iterator:
        x = []
        for t in valid_index:
            x.append(data[0][t])
        y = numpy.array(data[1])[valid_index]
        y2 = numpy.array(data[2])[valid_index]
	y3 = numpy.array(data[3])[valid_index]
	y4 = numpy.array(data[4])[valid_index]
        preds = f_pred(x)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
        if pr==True: 
            for i in len(valid_index):
                f.writelines([preds[i],' ',y[i],' ',y2[i],' ',y3[i],' ',y4[i],'\n'])
    if pr==True: 
        f.close()


    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])
    
    

    return valid_err

def train_lstm(
    patience=100,  # Number of epoch to wait before early stop if no progress
    max_epochs=200,  # The maximum number of epoch to run
    dispFreq=100,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.5,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=adadelta,  # sgd,adadelta and rmsprop available,sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=500,  # Compute the validation error after this number of update.
    saveFreq=500,  # Save the parameters after every saveFreq updates
    batch_size=50,  # The batch size during training.
    valid_batch_size=50,  # The batch size used for validation/test set.

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error# This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    valid_portion=0.05,
    test_portion=0.5,

    #DATA
    dataset='datapro',
    maxlen=128,  # Sequence longer then this get ignored
    data_x_length=6,  # Vocabulary size

    #CNN
    use_cnn = True,
    use_pooling = True,
    cnn_kernel_num=64,
    cnn_kernel_length=6,
    cnn_layer_num=2,
    pool_mode = 'max',

    #LSTM
    use_blstm = True,
    lstm_layer_num = 2,
    dim_proj=512,  # word embeding dimension and LSTM number of hidden units.
    ydim = 3,

    #DNN
    use_dnn = False,
    dnn_layer1_num = 512,
    dnn_layer2_num = 512,

    #result name
    result_name = 'resule.txt',

):

	# Model options
    model_options = locals().copy()
    print("model options", model_options)

    load_data = get_dataset(dataset)
    train, valid, test = load_data(valid_portion, test_portion, maxlen, data_x_length)
    #print('====================train y=====================',train[1])

    print('Building model')
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    tparams = init_tparams(params)
    (use_noise, x, y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay


    f_cost = theano.function([x, y], cost, name='f_cost')
    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads, x, y, cost)

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()


    f = open(result_name, 'a')

    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]
                
                n_samples += batch_size
               # print('pre===============:', f_pred_prob(x))
                cost = f_grad_shared(x, y)
                f_update(lrate)
                
                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    print("=============error====================")
                   # train_err = pred_error(f_pred, train, kf)
                    valid_err = pred_error(f_pred, valid,
                                           kf_valid)
                    history_errs.append([valid_err])

                    if (best_p is None or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print( ( 'Valid ', valid_err) )
                    f.writelines([ ' Valid: ', '%f' % valid_err])
                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, train, kf_train_sorted,False,'')
    valid_err = pred_error(f_pred, valid, kf_valid,False,'')
    test_err = pred_error(f_pred, test, kf_test, True, 'detail_result.txt')

    print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr)
    
 
    f.writelines(['====================start======================', '\n'])
    f.writelines(['use_cnn: ', '%d' % use_cnn, '\n'])
    f.writelines(['use_pooling: ', '%d' % use_pooling, '\n'])
    f.writelines(['cnn_kernel_num: ', '%d' % cnn_kernel_num, '\n'])
    f.writelines(['cnn_layer_num: ', '%d' % cnn_layer_num, '\n'])
    f.writelines(['pool_mode: ', '%s' % pool_mode, '\n'])
    f.writelines(['use_blstm: ', '%d' % use_blstm, '\n'])
    f.writelines(['lstm_layer_num: ', '%d' % lstm_layer_num, '\n'])
    f.writelines(['use_dnn: ', '%d' % use_dnn, '\n'])
    f.writelines(['dnn_layer1_num: ', '%d' % dnn_layer1_num, '\n'])
    f.writelines(['dnn_lauer2_num: ', '%d' % dnn_layer2_num, '\n'])
    f.writelines(['train_err: ', '%f' % train_err, '\n'])
    f.writelines(['valid_err: ', '%f' % valid_err, '\n'])
    f.writelines(['test_err: ', '%f' % test_err, '\n'])
    f.writelines(['====================end======================', '\n'])

    f.close()
    return train_err, valid_err, test_err

if __name__ == '__main__':
    # See function train for all possible parameter and there definit
    train_lstm(
        max_epochs=100,
        data_x_length=6,  
        #CNN
        use_cnn = True,
        use_pooling = False,
        cnn_kernel_num=64,
        cnn_kernel_length=6,
        cnn_layer_num=1,
        pool_mode = 'max',

        #LSTM
        use_blstm = True,
        lstm_layer_num = 2,

        #DNN
        use_dnn = True,
        dnn_layer1_num = 512,
        dnn_layer2_num = 512,

        #result_name
        result_name = 'result_6_0.8.txt',
        valid_portion=0.05,
        test_portion=0.2,
        saveto='lstm_model_6_0.8.npz',
    )


