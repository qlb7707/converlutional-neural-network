#================================================================
#   Copyright (C) 2016 All rights reserved.
#   
#   filename     :mycnn.py
#   author       :qinlibin
#   create date  :2016/03/19
#   mail         :qin_libin@foxmail.com
#
#================================================================

from theano.tensor.nnet import conv
import pylab
from PIL import Image
import numpy
import theano.tensor as T
import theano
rng = numpy.random.RandomState(23455)

input = T.tensor4(name='input')

w_shp = (2,3,9,9)
w_bound = numpy.sqrt(3*9*9)
W = theano.shared(numpy.asarray(rng.uniform(low = -1.0/ w_bound,high = \
        1.0 / w_bound,size = w_shp),dtype = input.dtype),name = 'W')
b_shp = (2,)
b = theano.shared(numpy.asarray(rng.uniform(low = -.5,high = .5,\
        size = b_shp),dtype = input.dtype),name = 'b')
conv_out =theano.tensor.nnet.conv.conv2d(input,W)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))
f = theano.function([input],output)
img = Image.open(open('images/3wolfmoon.jpg'))
img = numpy.asarray(img,dtype = 'float64') /256

img_ = img.swapaxes(0,2).swapaxes(1,2).reshape(1,3,639,516)
filtered_img = f(img_)

pylab.subplot(1,3,1);pylab.axis('off');pylab.imshow(img)
pylab.gray();

pylab.subplot(1,3,2);pylab.axis('off');
pylab.imshow(filtered_img[0,0,:,:])
pylab.subplot(1,3,3);pylab.axis('off');
pylab.imshow(filtered_img[0,1,:,:])
pylab.show()

