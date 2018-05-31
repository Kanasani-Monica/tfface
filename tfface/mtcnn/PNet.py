# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six import string_types, iteritems

import tensorflow as tf

from tfface.mtcnn.AbstractNeuralNetwork import AbstractNeuralNetwork

class PNet(AbstractNeuralNetwork):
    	def setup(self):
        	(self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             	     .conv(3, 3, 10, 1, 1, padding='VALID', relu=False, name='conv1')
             	     .prelu(name='PReLU1')
             	     .max_pool(2, 2, 2, 2, name='pool1')
             	     .conv(3, 3, 16, 1, 1, padding='VALID', relu=False, name='conv2')
             	     .prelu(name='PReLU2')
             	     .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv3')
             	     .prelu(name='PReLU3')
             	     .conv(1, 1, 2, 1, 1, relu=False, name='conv4-1')
             	     .softmax(3,name='prob1'))

        	(self.feed('PReLU3') #pylint: disable=no-value-for-parameter
             	     .conv(1, 1, 4, 1, 1, relu=False, name='conv4-2'))
