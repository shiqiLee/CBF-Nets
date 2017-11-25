from __future__ import print_function
from six.moves import xrange
import six.moves.cPickle as pickle

import gzip
import os

import numpy
import theano
import theano.tensor as tensor
from theano import config

def load_data(valid_portion=0.05, test_portion=0.2, step_time=128, xlength=6):
    train1, valid1, test1 = process_data(valid_portion, test_portion, step_time, 'data/101.txt', xlength)
    train2, valid2, test2 = process_data(valid_portion, test_portion, step_time, 'data/102.txt', xlength)
    train3, valid3, test3 = process_data(valid_portion, test_portion, step_time, 'data/103.txt', xlength)
    train4, valid4, test4 = process_data(valid_portion, test_portion, step_time, 'data/104.txt', xlength)
    train5, valid5, test5 = process_data(valid_portion, test_portion, step_time, 'data/105.txt', xlength)
    train6, valid6, test6 = process_data(valid_portion, test_portion, step_time, 'data/106.txt', xlength)
    train7, valid7, test7 = process_data(valid_portion, test_portion, step_time, 'data/107.txt', xlength)
    train8, valid8, test8 = process_data(valid_portion, test_portion, step_time, 'data/108.txt', xlength)
    train_set_x = train1[0] + train2[0] + train3[0] + train4[0] + train5[0] + train6[0] + train7[0] + train8[0]
    train_set_y = train1[1] + train2[1] + train3[1] + train4[1] + train5[1] + train6[1] + train7[1] + train8[1]
    train_set_y2 = train1[2] + train2[2] + train3[2] + train4[2] + train5[2] + train6[2] + train7[2] + train8[2]
    train_set_y3 = train1[3] + train2[3] + train3[3] + train4[3] + train5[3] + train6[3] + train7[3] + train8[3]
    train_set_y4 = train1[4] + train2[4] + train3[4] + train4[4] + train5[4] + train6[4] + train7[4] + train8[4]
    valid_set_x = valid1[0] + valid2[0] + valid3[0] + valid4[0] + valid5[0] + valid6[0] + valid7[0] + valid8[0]
    valid_set_y = valid1[1] + valid2[1] + valid3[1] + valid4[1] + valid5[1] + valid6[1] + valid7[1] + valid8[1]
    valid_set_y2 = valid1[2] + valid2[2] + valid3[2] + valid4[2] + valid5[2] + valid6[2] + valid7[2] + valid8[2]
    valid_set_y3 = valid1[3] + valid2[3] + valid3[3] + valid4[3] + valid5[3] + valid6[3] + valid7[3] + valid8[3]
    valid_set_y4 = valid1[4] + valid2[4] + valid3[4] + valid4[4] + valid5[4] + valid6[4] + valid7[4] + valid8[4]
    test_set_x = test1[0] + test2[0] + test3[0] + test4[0] + test5[0] + test6[0] + test7[0] + test8[0]
    test_set_y = test1[1] + test2[1] + test3[1] + test4[1] + test5[1] + test6[1] + test7[1] + test8[1]
    test_set_y2 = test1[2] + test2[2] + test3[2] + test4[2] + test5[2] + test6[2] + test7[2] + test8[2]
    test_set_y3 = test1[3] + test2[3] + test3[3] + test4[3] + test5[3] + test6[3] + test7[3] + test8[3]
    test_set_y4 = test1[4] + test2[4] + test3[4] + test4[4] + test5[4] + test6[4] + test7[4] + test8[4]
    
    train = (train_set_x, train_set_y, train_set_y2, train_set_y3, train_set_y4)
    valid = (valid_set_x, valid_set_y, valid_set_y2, valid_set_y3, valid_set_y4)
    test = (test_set_x, test_set_y, test_set_y2, test_set_y3, test_set_y4)
    print("train: ",len(train[0]))
    print("valid: ",len(valid[0]))
    print("test: ",len(test[0]))
    return train, valid, test


def process_data(valid_portion=0.05, test_portion=0.2, step_time=128, c='', xl=6):
    data = numpy.loadtxt(c)
    datasize= len(data)
    
    print('datasize:', datasize)
    trainnum = int(datasize*0.8)

    if xl == 6:
        set_x = data[:, 1:7] 
    else:
        set_x = data[:, 1:4]

    set_y = data[:, :1]
    set_y2 = data[:, 7:8]
    set_y4 = data[:, 8:]


    n_samples = len(set_x)/step_time - 1

    valid_set_x = []
    valid_set_y = []
    valid_set_y2 = []
    valid_set_y3 = []
    valid_set_y4 = []
    train_set_x = []
    train_set_y = []
    train_set_y2 = []
    train_set_y3 = []
    train_set_y4 = []
    test_set_x = []
    test_set_y = []
    test_set_y2 = []
    test_set_y3 = []
    test_set_y4 = []
 
    v = valid_portion
    t = test_portion
    
    for s in range(n_samples):
	if set_y[s*step_time] == set_y[(s+1)*step_time-1]:
	    if t >= 1:
	        test_set_x.append(set_x[(s*step_time):((s+1)*step_time)])
		test_set_y.append(set_y[s*step_time][0]-1)
		test_set_y2.append(set_y2[s*step_time][0])
		test_set_y3.append(set_y2[(s+1)*step_time-1][0])
		test_set_y4.append(set_y4[(s+1)*step_time-1][0])
		t = test_portion
	    else:
	        t = t + test_portion
                if  v >= 1:
		    valid_set_x.append(set_x[(s*step_time):((s+1)*step_time)])
		    valid_set_y.append(set_y[s*step_time][0]-1)
		    valid_set_y2.append(set_y2[s*step_time][0])
		    valid_set_y3.append(set_y2[(s+1)*step_time-1][0])
		    valid_set_y4.append(set_y4[(s+1)*step_time-1][0])
		    v = valid_portion
		if  v < 1:
	            train_set_x.append(set_x[(s*step_time):((s+1)*step_time)])
		    train_set_y.append(set_y[s*step_time][0]-1)
		    train_set_y2.append(set_y2[s*step_time][0])
	            train_set_y3.append(set_y2[(s+1)*step_time-1][0])
		    train_set_y4.append(set_y4[(s+1)*step_time-1][0])
		    v = v + valid_portion

    train = (train_set_x, train_set_y, train_set_y2, train_set_y3, train_set_y4)
    valid = (valid_set_x, valid_set_y, valid_set_y2, valid_set_y3, valid_set_y4)
    test = (test_set_x, test_set_y, test_set_y2, test_set_y3, test_set_y4)

    
    return train, valid, test

    

