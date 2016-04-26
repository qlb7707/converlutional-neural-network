#================================================================
#   Copyright (C) 2016 All rights reserved.
#   
#   filename     :gputest.py
#   author       :qinlibin
#   create date  :2016/03/21
#   mail         :qin_libin@foxmail.com
#
#================================================================

from theano import function, config, shared, tensor, sandbox 
import numpy
import time
vlen=10*30*768 #10x#coresx#threadspercore 
iters = 1000
rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX)) 
f = function([], tensor.exp(x)) 
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
        r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0)) 
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and\
    ('Gpu' not in type(x.op).__name__)\
    for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')

