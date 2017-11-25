from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy

if __name__ == '__main__':
	data = numpy.loadtxt('detail_result_6_0.8.txt')
	f = open('error_6_0.8.txt', 'a')
	l1 = [0,0,0]
	for i in range(len(data)):
		if data[i][0] != data[i][1]:
			f.writelines([' %d' % data[i][0], ' %d' % data[i][1], ' %d' % data[i][2], ' %d' % data[i][3], ' %d' % data[i][4],'\n'])

		l1[int(data[i][1])] = l1[int(data[i][1])]+1

	print(l1)

             


	f.close()
    
    
        
