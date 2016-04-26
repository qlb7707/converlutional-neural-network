#================================================================
#   Copyright (C) 2016 All rights reserved.
#   
#   filename     :cnn.py
#   author       :qinlibin
#   create date  :2016/03/23
#   mail         :qin_libin@foxmail.com
#
#================================================================

import os
import sys
import theano
from theano.tensor.nnet import conv
import cPickle
import time
import numpy
import theano.tensor as T

from logisticregression import LogisticRegression,load_data

from mlp import HiddenLayer

from theano.tensor.signal import downsample

class LeConvPoolLayer(object):
    def __init__(self,rng,input,filter_shape,image_shape,poolsize =(2,2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = numpy.prod(filter_shape[1:])
        W_values = numpy.asarray(rng.uniform(low = -numpy.sqrt(3./fan_in),\
               high = numpy.sqrt(3./fan_in),size = filter_shape),dtype=\
               theano.config.floatX)
        self.W = theano.shared(value=W_values,name='W')
        b_values = numpy.zeros((filter_shape[0],),dtype=theano.config.floatX)        
        self.b = theano.shared(value=b_values,name='b')
        conv_out = conv.conv2d(input,self.W,filter_shape = filter_shape,image_shape=image_shape)
        pooled_out = downsample.max_pool_2d(conv_out,poolsize,ignore_border=True)
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))
        self.params = [self.W,self.b]



def testLeNet(learning_rate = 0.01,n_epochs = 1000,batch_size = 20,n_hidden = 500,dataset='mnist.pkl.gz'):

    datasets = load_data(dataset)
    train_set_x,train_set_y = datasets[0]
    valid_set_x,valid_set_y = datasets[1]
    test_set_x,test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print '...building the model'


    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(1234)
    layer0_input = x.reshape((batch_size,1,28,28))
    layer0 = LeConvPoolLayer(rng,input = layer0_input,image_shape=\
            (batch_size,1,28,28),filter_shape=(20,1,5,5),\
            poolsize=(2,2))
    layer1 = LeConvPoolLayer(rng,input=layer0.output,image_shape=\
            (batch_size,20,12,12),filter_shape=(50,20,5,5),poolsize=(2,2))
    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(rng,input=layer2_input,n_in=50*4*4,n_out=500,\
            activation = T.tanh)
    layer3 = LogisticRegression(input=layer2.output,n_in=500,n_out=10)
    cost = layer3.negative_log_likelihood(y)
    test_model = theano.function(inputs=[index],outputs=layer3.errors(y),\
            givens={x:test_set_x[index*batch_size:(index+1)*batch_size],\
            y:test_set_y[index*batch_size:(index+1)*batch_size]})
    params = layer3.params + layer2.params + layer1.params + layer0.params
    grads = T.grad(cost,params)
    updates = []
    for param_i,grad_i in zip(params,grads):
        updates.append((param_i,param_i - learning_rate * grad_i))
    train_model = theano.function([index],cost,updates=updates,\
            givens={x:train_set_x[index*batch_size:(index+1)*batch_size],\
            y:train_set_y[index*batch_size:(index+1)*batch_size]})

    validate_model = theano.function(inputs=[index],outputs=layer3.errors(y),\
            givens={x:valid_set_x[index*batch_size:(index+1)*batch_size],\
            y:valid_set_y[index*batch_size:(index+1)*batch_size]})

    print '...training'

    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches,patience/2)

    best_param = None
    best_validation_loss = numpy.inf

    best_iter = 0
    best_score = numpy.inf

    
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1)*n_train_batches + minibatch_index
            if(iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print ('epoch %i minibatch %i/%i,validation error %f %%'%\
                        (epoch,minibatch_index+1,n_train_batches,this_validation_loss*100.))
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience,iter*patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print('     epoch %i,minibatch %i/%i,test error of best model %f %%')%\
                            (epoch,minibatch_index+1,n_train_batches,test_score*100.)


        if patience <= iter:
            done_looping = True
            break
    end_time = time.clock()
    print(('optimization complete.best validation score of %f %%'\
            'obtained at iteration %i with test performance %f %%')%\
            (best_validation_loss*100,best_iter+1,test_score*100.))

    print >> sys.stderr,('the code for file' + os.path.split(__file__)[1]\
            +' ran for %.2fm' % ((end_time-start_time)/60.))




    
    
if __name__ == '__main__':
    testLeNet()
